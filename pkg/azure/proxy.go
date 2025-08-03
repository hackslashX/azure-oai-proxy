package azure

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"strings"
	"encoding/json"

	"github.com/tidwall/gjson"
)

var (
	AzureOpenAIAPIVersion       = "2024-12-01-preview" // API version for proxying requests
	AzureOpenAIModelsAPIVersion = "2024-10-21"         // API version for fetching models
	AzureOpenAIResponsesAPIVersion = "preview"           // API version for Responses API
	AzureOpenAIEndpoint         = ""
	ServerlessDeploymentInfo    = make(map[string]ServerlessDeployment)
	AzureOpenAIModelMapper      = make(map[string]string)
)

type ServerlessDeployment struct {
	Name   string
	Region string
	Key    string
}

func init() {
	if v := os.Getenv("AZURE_OPENAI_APIVERSION"); v != "" {
		AzureOpenAIAPIVersion = v
	}
	if v := os.Getenv("AZURE_OPENAI_MODELS_APIVERSION"); v != "" {
		AzureOpenAIModelsAPIVersion = v
	}
	if v := os.Getenv("AZURE_OPENAI_ENDPOINT"); v != "" {
		AzureOpenAIEndpoint = v
	}

	if v := os.Getenv("AZURE_AI_STUDIO_DEPLOYMENTS"); v != "" {
		for _, pair := range strings.Split(v, ",") {
			info := strings.Split(pair, "=")
			if len(info) == 2 {
				deploymentInfo := strings.Split(info[1], ":")
				if len(deploymentInfo) == 2 {
					ServerlessDeploymentInfo[strings.ToLower(info[0])] = ServerlessDeployment{
						Name:   deploymentInfo[0],
						Region: deploymentInfo[1],
						Key:    os.Getenv("AZURE_OPENAI_KEY_" + strings.ToUpper(info[0])),
					}
				}
			}
		}
	}

	// Initialize AzureOpenAIModelMapper with updated model list and hardcode as failsafe
	AzureOpenAIModelMapper = map[string]string{
		"o1-preview":                  "o1-preview",
		"o1-mini-2024-09-12":          "o1-mini-2024-09-12",
		"gpt-4o":                      "gpt-4o",
		"gpt-4o-2024-05-13":           "gpt-4o-2024-05-13",
		"gpt-4o-2024-08-06":           "gpt-4o-2024-08-06",
		"gpt-4o-mini":                 "gpt-4o-mini",
		"gpt-4o-mini-2024-07-18":      "gpt-4o-mini-2024-07-18",
		"gpt-4":                       "gpt-4-0613",
		"gpt-4-0613":                  "gpt-4-0613",
		"gpt-4-1106-preview":          "gpt-4-1106-preview",
		"gpt-4-0125-preview":          "gpt-4-0125-preview",
		"gpt-4-vision-preview":        "gpt-4-vision-preview",
		"gpt-4-turbo-2024-04-09":      "gpt-4-turbo-2024-04-09",
		"gpt-4-32k":                   "gpt-4-32k-0613",
		"gpt-4-32k-0613":              "gpt-4-32k-0613",
		"gpt-3.5-turbo":               "gpt-35-turbo-0613",
		"gpt-3.5-turbo-0301":          "gpt-35-turbo-0301",
		"gpt-3.5-turbo-0613":          "gpt-35-turbo-0613",
		"gpt-3.5-turbo-1106":          "gpt-35-turbo-1106",
		"gpt-3.5-turbo-0125":          "gpt-35-turbo-0125",
		"gpt-3.5-turbo-16k":           "gpt-35-turbo-16k-0613",
		"gpt-3.5-turbo-16k-0613":      "gpt-35-turbo-16k-0613",
		"gpt-3.5-turbo-instruct":      "gpt-35-turbo-instruct-0914",
		"gpt-3.5-turbo-instruct-0914": "gpt-35-turbo-instruct-0914",
		"text-embedding-3-small":      "text-embedding-3-small-1",
		"text-embedding-3-large":      "text-embedding-3-large-1",
		"text-embedding-ada-002":      "text-embedding-ada-002-2",
		"text-embedding-ada-002-1":    "text-embedding-ada-002-1",
		"text-embedding-ada-002-2":    "text-embedding-ada-002-2",
		"dall-e-2":                    "dall-e-2-2.0",
		"dall-e-2-2.0":                "dall-e-2-2.0",
		"dall-e-3":                    "dall-e-3-3.0",
		"dall-e-3-3.0":                "dall-e-3-3.0",
		"babbage-002":                 "babbage-002-1",
		"babbage-002-1":               "babbage-002-1",
		"davinci-002":                 "davinci-002-1",
		"davinci-002-1":               "davinci-002-1",
		"tts":                         "tts-001",
		"tts-001":                     "tts-001",
		"tts-hd":                      "tts-hd-001",
		"tts-hd-001":                  "tts-hd-001",
		"whisper":                     "whisper-001",
		"whisper-001":                 "whisper-001",
	}

	log.Printf("Loaded ServerlessDeploymentInfo: %+v", ServerlessDeploymentInfo)
	log.Printf("Azure OpenAI Endpoint: %s", AzureOpenAIEndpoint)
	log.Printf("Azure OpenAI API Version: %s", AzureOpenAIAPIVersion)
	log.Printf("Azure OpenAI Models API Version: %s", AzureOpenAIModelsAPIVersion)
}

func NewOpenAIReverseProxy() *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director:       makeDirector(),
		ModifyResponse: modifyResponse,
	}
}

func HandleToken(req *http.Request) {
	model := getModelFromRequest(req)
	modelLower := strings.ToLower(model)
	// Check if it's a serverless deployment
	if info, ok := ServerlessDeploymentInfo[modelLower]; ok {
		// Set the correct authorization header for serverless
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", info.Key))
		req.Header.Del("api-key")
		log.Printf("Using serverless deployment authentication for %s", model)
	} else {
		// For regular Azure OpenAI deployments, use the api-key
		apiKey := req.Header.Get("api-key")
		if apiKey == "" {
			apiKey = req.Header.Get("Authorization")
			if strings.HasPrefix(apiKey, "Bearer ") {
				apiKey = strings.TrimPrefix(apiKey, "Bearer ")
			}
		}
		if apiKey == "" {
			log.Printf("Warning: No api-key or Authorization header found for deployment: %s", model)
		} else {
			req.Header.Set("api-key", apiKey)
			req.Header.Del("Authorization")
			log.Printf("Using regular Azure OpenAI authentication for %s", model)
		}
	}
}

func makeDirector() func(*http.Request) {
    return func(req *http.Request) {
        model := getModelFromRequest(req)
        originURL := req.URL.String()
        log.Printf("Original request URL: %s for model: %s", originURL, model)

        // Check if this is a chat completion request for a model that should use Responses API
        if strings.HasPrefix(req.URL.Path, "/v1/chat/completions") && shouldUseResponsesAPI(model) {
            log.Printf("Redirecting %s from chat/completions to responses API", model)
            // Convert the chat completion request to a responses request
            convertChatToResponses(req)
        }

        // Handle the token
        HandleToken(req)

        // Convert model to lowercase for case-insensitive matching
        modelLower := strings.ToLower(model)

        // Check if it's a serverless deployment
        if info, ok := ServerlessDeploymentInfo[modelLower]; ok {
            handleServerlessRequest(req, info, model)
        } else if azureModel, ok := AzureOpenAIModelMapper[modelLower]; ok {
            handleRegularRequest(req, azureModel)
        } else {
            log.Printf("Warning: Unknown model %s, treating as regular Azure OpenAI deployment", model)
            handleRegularRequest(req, model)
        }

        log.Printf("Proxying request [%s] %s -> %s", model, originURL, req.URL.String())
    }
}

func handleServerlessRequest(req *http.Request, info ServerlessDeployment, model string) {
	req.URL.Scheme = "https"
	req.URL.Host = fmt.Sprintf("%s.%s.models.ai.azure.com", info.Name, info.Region)
	req.Host = req.URL.Host // Preserve query parameters from the original request
	originalQuery := req.URL.Query()
	for key, values := range originalQuery {
		for _, value := range values {
			req.URL.Query().Add(key, value)
		}
	}

	// Set the correct authorization header for serverless
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", info.Key))
	req.Header.Del("api-key")
	log.Printf("Using serverless deployment for %s", model)
}

func handleRegularRequest(req *http.Request, deployment string) {
    remote, _ := url.Parse(AzureOpenAIEndpoint)
    req.URL.Scheme = remote.Scheme
    req.URL.Host = remote.Host
    req.Host = remote.Host

    // Handle Responses API endpoints
    if strings.Contains(req.URL.Path, "/v1/responses") {
        // For Responses API, we need to handle the paths differently
        if strings.HasPrefix(req.URL.Path, "/v1/responses") && !strings.Contains(req.URL.Path, "/responses/") {
            // POST /v1/responses - Create response
            req.URL.Path = "/openai/v1/responses"
        } else {
            // Other responses endpoints (GET, DELETE, etc.)
            // Convert /v1/responses/{id} to /openai/v1/responses/{id}
            req.URL.Path = strings.Replace(req.URL.Path, "/v1/", "/openai/v1/", 1)
        }
        
        // Use the preview API version for Responses API
        query := req.URL.Query()
        query.Set("api-version", AzureOpenAIResponsesAPIVersion)
        req.URL.RawQuery = query.Encode()
    } else {
        // Existing logic for other endpoints
        switch {
        case strings.HasPrefix(req.URL.Path, "/v1/chat/completions"):
            req.URL.Path = path.Join("/openai/deployments", deployment, "chat/completions")
        case strings.HasPrefix(req.URL.Path, "/v1/completions"):
            req.URL.Path = path.Join("/openai/deployments", deployment, "completions")
        case strings.HasPrefix(req.URL.Path, "/v1/embeddings"):
            req.URL.Path = path.Join("/openai/deployments", deployment, "embeddings")
        case strings.HasPrefix(req.URL.Path, "/v1/images/generations"):
            req.URL.Path = path.Join("/openai/deployments", deployment, "images/generations")
        case strings.HasPrefix(req.URL.Path, "/v1/audio/"):
            // Handle audio endpoints
            audioPath := strings.TrimPrefix(req.URL.Path, "/v1/")
            req.URL.Path = path.Join("/openai/deployments", deployment, audioPath)
        case strings.HasPrefix(req.URL.Path, "/v1/files"):
            // Files API doesn't use deployment in path
            req.URL.Path = strings.Replace(req.URL.Path, "/v1/", "/openai/", 1)
        default:
            req.URL.Path = path.Join("/openai/deployments", deployment, strings.TrimPrefix(req.URL.Path, "/v1/"))
        }
        
        // Add api-version query parameter for non-Responses API
        query := req.URL.Query()
        query.Add("api-version", AzureOpenAIAPIVersion)
        req.URL.RawQuery = query.Encode()
    }

    // Use the api-key from the original request for regular deployments
    apiKey := req.Header.Get("api-key")
    if apiKey == "" {
        log.Printf("Warning: No api-key found for regular deployment: %s", deployment)
    }
    log.Printf("Using regular Azure OpenAI deployment for %s", deployment)
}

func getModelFromRequest(req *http.Request) string {
    // For Responses API, always check the body first
    if strings.Contains(req.URL.Path, "/responses") && req.Body != nil {
        body, _ := io.ReadAll(req.Body)
        req.Body = io.NopCloser(bytes.NewBuffer(body))
        
        // The Responses API uses "model" field in the request body
        model := gjson.GetBytes(body, "model").String()
        if model != "" {
            return model
        }
    }
    
    // Existing logic for path-based model detection
    parts := strings.Split(req.URL.Path, "/")
    for i, part := range parts {
        if part == "deployments" && i+1 < len(parts) {
            return parts[i+1]
        }
    }

    // If not found in the path, try to get it from the request body
    if req.Body != nil {
        body, _ := io.ReadAll(req.Body)
        req.Body = io.NopCloser(bytes.NewBuffer(body))
        model := gjson.GetBytes(body, "model").String()
        if model != "" {
            return model
        }
    }

    // If still not found, return an empty string
    return ""
}

func sanitizeHeaders(headers http.Header) http.Header {
	sanitized := make(http.Header)
	for key, values := range headers {
		if key == "Authorization" || key == "api-key" {
			sanitized[key] = []string{"[REDACTED]"}
		} else {
			sanitized[key] = values
		}
	}
	return sanitized
}

func modifyResponse(res *http.Response) error {
    // Check if this is a Responses API response that needs conversion
    if res.Request.URL.Path == "/openai/v1/responses" && res.StatusCode == 200 {
        // Check if the original request was for chat completions
        if origPath := res.Request.Header.Get("X-Original-Path"); origPath == "/v1/chat/completions" {
            convertResponsesToChatCompletion(res)
        }
    }

    if res.StatusCode >= 400 {
        body, _ := io.ReadAll(res.Body)
        log.Printf("Azure API Error Response: Status: %d, Body: %s", res.StatusCode, string(body))
        res.Body = io.NopCloser(bytes.NewBuffer(body))
    }
    
    if res.Header.Get("Content-Type") == "text/event-stream" {
        res.Header.Set("X-Accel-Buffering", "no")
        res.Header.Set("Cache-Control", "no-cache")
        res.Header.Set("Connection", "keep-alive")
    }
    
    return nil
}

// Add a function to check if a model should use Responses API
func shouldUseResponsesAPI(model string) bool {
    modelLower := strings.ToLower(model)
    // Models that should use Responses API instead of chat completions
    responsesModels := []string{
        "o3", "o3-pro", "o3-mini", "o4", "o4-mini", "o1", "o1-preview", "o1-mini",
    }
    
    for _, m := range responsesModels {
        if strings.HasPrefix(modelLower, m) {
            return true
        }
    }
    return false
}

// Function to convert chat completion request to responses format
func convertChatToResponses(req *http.Request) {
    if req.Body != nil {
        body, _ := io.ReadAll(req.Body)
        
        // Parse the chat completion request
        messages := gjson.GetBytes(body, "messages").Array()
        temperature := gjson.GetBytes(body, "temperature").Float()
        maxTokens := gjson.GetBytes(body, "max_tokens").Int()
        stream := gjson.GetBytes(body, "stream").Bool()
        
        // Convert messages to input format for Responses API
        var input []map[string]interface{}
        for _, msg := range messages {
            role := msg.Get("role").String()
            content := msg.Get("content").String()
            
            inputMsg := map[string]interface{}{
                "role": role,
                "content": []map[string]interface{}{
                    {
                        "type": "input_text",
                        "text": content,
                    },
                },
            }
            input = append(input, inputMsg)
        }
        
        // Create new request body for Responses API
        newBody := map[string]interface{}{
            "model": gjson.GetBytes(body, "model").String(),
            "input": input,
        }
        
        if temperature > 0 {
            newBody["temperature"] = temperature
        }
        if maxTokens > 0 {
            newBody["max_output_tokens"] = maxTokens
        }
        if stream {
            newBody["stream"] = true
        }
        
        // Marshal the new body
        newBodyBytes, _ := json.Marshal(newBody)
        req.Body = io.NopCloser(bytes.NewBuffer(newBodyBytes))
        req.ContentLength = int64(len(newBodyBytes))
        
        // Update the path to use responses endpoint
        req.URL.Path = "/v1/responses"
    }
}

// Add function to convert Responses API response to chat completion format
func convertResponsesToChatCompletion(res *http.Response) {
    body, _ := io.ReadAll(res.Body)
    
    var responseData map[string]interface{}
    json.Unmarshal(body, &responseData)
    
    // Extract the output
    outputs := responseData["output"].([]interface{})
    var content string
    
    for _, output := range outputs {
        outputMap := output.(map[string]interface{})
        if outputMap["type"] == "message" && outputMap["role"] == "assistant" {
            contents := outputMap["content"].([]interface{})
            for _, c := range contents {
                contentMap := c.(map[string]interface{})
                if contentMap["type"] == "output_text" {
                    content = contentMap["text"].(string)
                    break
                }
            }
        }
    }
    
    // Create chat completion response
    chatResponse := map[string]interface{}{
        "id": responseData["id"],
        "object": "chat.completion",
        "created": int64(responseData["created_at"].(float64)),
        "model": responseData["model"],
        "choices": []map[string]interface{}{
            {
                "index": 0,
                "message": map[string]interface{}{
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            },
        },
        "usage": responseData["usage"],
    }
    
    // Marshal and set as new body
    newBody, _ := json.Marshal(chatResponse)
    res.Body = io.NopCloser(bytes.NewBuffer(newBody))
    res.ContentLength = int64(len(newBody))
    res.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))
}