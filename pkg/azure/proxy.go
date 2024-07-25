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
	AzureOpenAIAPIVersion    = "2024-06-01"
	AzureOpenAIEndpoint      = ""
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
	log.Printf("Azure OpenAI Endpoint: %s", AzureOpenAIEndpoint)
	log.Printf("Azure OpenAI API Version: %s", AzureOpenAIAPIVersion)
}

func NewOpenAIReverseProxy() *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director:       makeDirector(),
		ModifyResponse: modifyResponse,
	}
}

func makeDirector() func(*http.Request) {
	return func(req *http.Request) {
		model := getModelFromRequest(req)
		originURL := req.URL.String()
		log.Printf("Original request URL: %s for model: %s", originURL, model)

		// Check if it's a serverless deployment
		if info, ok := ServerlessDeploymentInfo[model]; ok {
			handleServerlessRequest(req, info, model)
		} else {
			handleRegularRequest(req, model)
		}

		log.Printf("Proxying request [%s] %s -> %s", model, originURL, req.URL.String())
		log.Printf("Final request headers: %v", sanitizeHeaders(req.Header))
	}
}

func handleServerlessRequest(req *http.Request, info ServerlessDeployment, model string) {
	req.URL.Scheme = "https"
	req.URL.Host = fmt.Sprintf("%s.%s.models.ai.azure.com", info.Name, info.Region)
	req.Host = req.URL.Host

	// Keep the original path for serverless deployments
	// req.URL.Path remains unchanged

	// Set the correct authorization header for serverless
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", info.Key))
	req.Header.Del("api-key")

	log.Printf("Using serverless deployment for %s", model)
}

func handleRegularRequest(req *http.Request, model string) {
	remote, _ := url.Parse(AzureOpenAIEndpoint)
	req.URL.Scheme = remote.Scheme
	req.URL.Host = remote.Host
	req.Host = remote.Host

	// Construct the path for regular Azure OpenAI deployments
	deployment := model // Use the model as the deployment name for regular deployments
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
		log.Printf("Warning: No api-key found for regular deployment: %s", model)
	}

	log.Printf("Using regular Azure OpenAI deployment for %s", model)
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
