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

	"github.com/tidwall/gjson"
)

var (
	AzureOpenAIAPIVersion       = "2024-12-01-preview" // API version for proxying requests
	AzureOpenAIModelsAPIVersion = "2024-10-21"         // API version for fetching models
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
		// log.Printf("Final request headers: %v", sanitizeHeaders(req.Header))
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

	// Construct the path for regular Azure OpenAI deployments
	switch {
	case strings.HasPrefix(req.URL.Path, "/v1/chat/completions"):
		req.URL.Path = path.Join("/openai/deployments", deployment, "chat/completions")
	case strings.HasPrefix(req.URL.Path, "/v1/completions"):
		req.URL.Path = path.Join("/openai/deployments", deployment, "completions")
	case strings.HasPrefix(req.URL.Path, "/v1/embeddings"):
		req.URL.Path = path.Join("/openai/deployments", deployment, "embeddings")
		// Add other cases as needed
	default:
		req.URL.Path = path.Join("/openai/deployments", deployment, strings.TrimPrefix(req.URL.Path, "/v1/"))
	}

	// Add api-version query parameter
	query := req.URL.Query()
	query.Add("api-version", AzureOpenAIAPIVersion)
	req.URL.RawQuery = query.Encode()

	// Use the api-key from the original request for regular deployments
	apiKey := req.Header.Get("api-key")
	if apiKey == "" {
		log.Printf("Warning: No api-key found for regular deployment: %s", deployment)
	}
	log.Printf("Using regular Azure OpenAI deployment for %s", deployment)
}

func getModelFromRequest(req *http.Request) string {
	// First, try to get the model from the URL path
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
	if res.StatusCode >= 400 {
		body, _ := io.ReadAll(res.Body)
		log.Printf("Azure API Error Response: Status: %d, Body: %s", res.StatusCode, string(body))
		res.Body = io.NopCloser(bytes.NewBuffer(body))
	}
	if res.Header.Get("Content-Type") == "text/event-stream" {
		res.Header.Set("X-Accel-Buffering", "no")
	}
	return nil
}
