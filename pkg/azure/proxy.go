package azure

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"regexp"
	"strings"

	"github.com/tidwall/gjson"
)

var (
	AzureOpenAIToken       = ""
	AzureOpenAIAPIVersion  = "2024-06-01"
	AzureOpenAIEndpoint    = ""
	AzureOpenAIModelMapper = map[string]string{
		"gpt-3.5-turbo":               "gpt-35-turbo",
		"gpt-3.5-turbo-0125":          "gpt-35-turbo-0125",
		"gpt-3.5-turbo-0613":          "gpt-35-turbo-0613",
		"gpt-3.5-turbo-1106":          "gpt-35-turbo-1106",
		"gpt-3.5-turbo-16k-0613":      "gpt-35-turbo-16k-0613",
		"gpt-3.5-turbo-instruct-0914": "gpt-35-turbo-instruct-0914",
		"gpt-4":                       "gpt-4-0613",
		"gpt-4-32k":                   "gpt-4-32k",
		"gpt-4-32k-0613":              "gpt-4-32k-0613",
		"gpt-4o":                      "gpt-4o",
		"gpt-4o-mini":                 "gpt-4o-mini",
		"gpt-4o-2024-05-13":           "gpt-4o-2024-05-13",
		"gpt-4-turbo":                 "gpt-4-turbo",
		"gpt-4-vision-preview":        "gpt-4-vision-preview",
		"gpt-4-turbo-2024-04-09":      "gpt-4-turbo-2024-04-09",
		"gpt-4-1106-preview":          "gpt-4-1106-preview",
		"text-embedding-ada-002":      "text-embedding-ada-002",
		"dall-e-2":                    "dall-e-2",
		"dall-e-3":                    "dall-e-3",
		"babbage-002":                 "babbage-002",
		"davinci-002":                 "davinci-002",
		"whisper-1":                   "whisper",
		"tts-1":                       "tts",
		"tts-1-hd":                    "tts-hd",
		"text-embedding-3-small":      "text-embedding-3-small-1",
		"text-embedding-3-large":      "text-embedding-3-large-1",
	}
	AzureAIStudioDeployments = make(map[string]string)
	fallbackModelMapper      = regexp.MustCompile(`[.:]`)
	ServerlessDeploymentInfo = make(map[string]ServerlessDeployment)
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
	if v := os.Getenv("AZURE_OPENAI_ENDPOINT"); v != "" {
		AzureOpenAIEndpoint = v
	}

	handleModelMapper()

	if v := os.Getenv("AZURE_AI_STUDIO_DEPLOYMENTS"); v != "" {
		for _, pair := range strings.Split(v, ",") {
			info := strings.Split(pair, "=")
			if len(info) == 2 {
				deploymentInfo := strings.Split(info[1], ":")
				if len(deploymentInfo) == 2 {
					ServerlessDeploymentInfo[info[0]] = ServerlessDeployment{
						Name:   deploymentInfo[0],
						Region: deploymentInfo[1],
						Key:    os.Getenv("AZURE_OPENAI_KEY_" + strings.ToUpper(info[0])),
					}
				}
			}
		}
	}
	log.Printf("Loaded ServerlessDeploymentInfo: %+v", ServerlessDeploymentInfo)

	if v := os.Getenv("AZURE_OPENAI_TOKEN"); v != "" {
		AzureOpenAIToken = v
		log.Printf("loading azure api token from env")
	}

	log.Printf("loading azure api endpoint: %s", AzureOpenAIEndpoint)
	log.Printf("loading azure api version: %s", AzureOpenAIAPIVersion)
	for k, v := range AzureOpenAIModelMapper {
		log.Printf("final azure model mapper: %s -> %s", k, v)
	}
	for k, v := range AzureAIStudioDeployments {
		log.Printf("loading azure ai studio deployment: %s -> %s", k, v)
	}
	log.Printf("Loaded %d serverless deployment infos", len(ServerlessDeploymentInfo))
}

func setServerlessAuth(req *http.Request, info ServerlessDeployment, deployment string) string {
	token := fmt.Sprintf("Bearer %s", info.Key)
	req.Header.Set("Authorization", token)
	req.Header.Del("api-key")
	log.Printf("Using serverless deployment authentication for %s", deployment)
	return deployment // Return the actual deployment name
}

func handleRegularAuth(req *http.Request, deployment string) string {
	var token string
	if apiKey := req.Header.Get("api-key"); apiKey != "" {
		token = apiKey
	} else if authHeader := req.Header.Get("Authorization"); authHeader != "" {
		token = strings.TrimPrefix(authHeader, "Bearer ")
	} else if AzureOpenAIToken != "" {
		token = AzureOpenAIToken
	} else if envApiKey := os.Getenv("AZURE_OPENAI_API_KEY"); envApiKey != "" {
		token = envApiKey
	}

	if token != "" {
		req.Header.Set("api-key", token)
		req.Header.Del("Authorization")
		log.Printf("Using regular Azure OpenAI authentication for %s", deployment)
	} else {
		log.Printf("Warning: No authentication token found for deployment: %s", deployment)
	}
	return deployment
}

func handleModelMapper() {
	overrideMode := strings.ToLower(os.Getenv("AZURE_OPENAI_MODEL_MAPPER_MODE")) == "override"

	if v := os.Getenv("AZURE_OPENAI_MODEL_MAPPER"); v != "" {
		for _, pair := range strings.Split(v, ",") {
			info := strings.Split(pair, "=")
			if len(info) == 2 {
				if overrideMode {
					AzureOpenAIModelMapper[info[0]] = info[1]
					log.Printf("Overriding model mapping: %s -> %s", info[0], info[1])
				} else {
					if _, exists := AzureOpenAIModelMapper[info[0]]; !exists {
						AzureOpenAIModelMapper[info[0]] = info[1]
						log.Printf("Adding new model mapping: %s -> %s", info[0], info[1])
					} else {
						log.Printf("Skipping existing model mapping: %s", info[0])
					}
				}
			} else {
				log.Printf("error parsing AZURE_OPENAI_MODEL_MAPPER, invalid value %s", pair)
			}
		}
	}
}

func NewOpenAIReverseProxy() *httputil.ReverseProxy {
	remote, err := url.Parse(AzureOpenAIEndpoint)
	if err != nil {
		log.Printf("error parse endpoint: %s\n", AzureOpenAIEndpoint)
		os.Exit(1)
	}

	return &httputil.ReverseProxy{
		Director:       makeDirector(remote),
		ModifyResponse: modifyResponse,
	}
}

func getModelFromRequest(req *http.Request) string {
	if req.Body == nil {
		return ""
	}
	body, _ := io.ReadAll(req.Body)
	req.Body = io.NopCloser(bytes.NewBuffer(body))
	return gjson.GetBytes(body, "model").String()
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

func extractDeploymentFromPath(path string) string {
	parts := strings.Split(path, "/")
	for i, part := range parts {
		if part == "deployments" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	return ""
}

func HandleToken(req *http.Request) string {
	deployment := extractDeploymentFromPath(req.URL.Path)

	// First, try an exact match for serverless deployment
	if info, ok := ServerlessDeploymentInfo[deployment]; ok {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", info.Key))
		req.Header.Del("api-key")
		log.Printf("Using serverless deployment authentication for %s", deployment)
		return deployment
	}

	// If no serverless match, proceed with regular Azure OpenAI authentication
	var token string
	if apiKey := req.Header.Get("api-key"); apiKey != "" {
		token = apiKey
	} else if authHeader := req.Header.Get("Authorization"); authHeader != "" {
		token = strings.TrimPrefix(authHeader, "Bearer ")
	} else if AzureOpenAIToken != "" {
		token = AzureOpenAIToken
	} else if envApiKey := os.Getenv("AZURE_OPENAI_API_KEY"); envApiKey != "" {
		token = envApiKey
	}

	if token != "" {
		req.Header.Set("api-key", token)
		req.Header.Del("Authorization")
		log.Printf("Using regular Azure OpenAI authentication for %s", deployment)
	} else {
		log.Printf("Warning: No authentication token found for deployment: %s", deployment)
	}
	return deployment
}

func makeDirector(remote *url.URL) func(*http.Request) {
	return func(req *http.Request) {
		model := getModelFromRequest(req)
		deployment := HandleToken(req)

		originURL := req.URL.String()
		log.Printf("Original request URL: %s for model: %s", originURL, model)

		if info, ok := ServerlessDeploymentInfo[deployment]; ok {
			req.URL.Scheme = "https"
			req.URL.Host = fmt.Sprintf("%s.%s.models.ai.azure.com", info.Name, info.Region)
			req.Host = req.URL.Host
			// For serverless, keep the original path
			log.Printf("Using serverless deployment for %s", deployment)
		} else {
			req.URL.Scheme = remote.Scheme
			req.URL.Host = remote.Host
			req.Host = remote.Host

			// For regular Azure OpenAI, construct the path
			switch {
			case strings.HasPrefix(req.URL.Path, "/v1/chat/completions"):
				req.URL.Path = path.Join("/openai/deployments", deployment, "chat/completions")
			case strings.HasPrefix(req.URL.Path, "/v1/completions"):
				req.URL.Path = path.Join("/openai/deployments", deployment, "completions")
			case strings.HasPrefix(req.URL.Path, "/v1/embeddings"):
				req.URL.Path = path.Join("/openai/deployments", deployment, "embeddings")
			// ... (keep other cases)
			default:
				req.URL.Path = path.Join("/openai/deployments", deployment, strings.TrimPrefix(req.URL.Path, "/v1/"))
			}

			// Only add api-version for non-serverless deployments
			query := req.URL.Query()
			query.Add("api-version", AzureOpenAIAPIVersion)
			req.URL.RawQuery = query.Encode()
		}

		req.URL.RawPath = req.URL.EscapedPath()

		if req.Body != nil {
			var requestBody map[string]interface{}
			bodyBytes, _ := io.ReadAll(req.Body)
			json.Unmarshal(bodyBytes, &requestBody)

			newParams := []string{"completion_config", "presence_penalty", "frequency_penalty", "best_of"}
			for _, param := range newParams {
				if val, ok := requestBody[param]; ok {
					log.Printf("Request includes %s parameter: %v", param, val)
				}
			}

			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		}

		log.Printf("Proxying request [%s] %s -> %s", model, originURL, req.URL.String())
		log.Printf("Final request headers: %v", sanitizeHeaders(req.Header))
	}
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

func GetDeploymentByModel(model string) string {
	if v, ok := AzureOpenAIModelMapper[model]; ok {
		return v
	}
	return fallbackModelMapper.ReplaceAllString(model, "")
}
