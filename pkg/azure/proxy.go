package azure

import (
	"bytes"
	"encoding/json"
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
	fallbackModelMapper = regexp.MustCompile(`[.:]`)
)

func init() {
	if v := os.Getenv("AZURE_OPENAI_APIVERSION"); v != "" {
		AzureOpenAIAPIVersion = v
	}
	if v := os.Getenv("AZURE_OPENAI_ENDPOINT"); v != "" {
		AzureOpenAIEndpoint = v
	}
	if v := os.Getenv("AZURE_OPENAI_MODEL_MAPPER"); v != "" {
		for _, pair := range strings.Split(v, ",") {
			info := strings.Split(pair, "=")
			if len(info) != 2 {
				log.Printf("error parsing AZURE_OPENAI_MODEL_MAPPER, invalid value %s", pair)
				os.Exit(1)
			}
			AzureOpenAIModelMapper[info[0]] = info[1]
		}
	}
	if v := os.Getenv("AZURE_OPENAI_TOKEN"); v != "" {
		AzureOpenAIToken = v
		log.Printf("loading azure api token from env")
	}

	log.Printf("loading azure api endpoint: %s", AzureOpenAIEndpoint)
	log.Printf("loading azure api version: %s", AzureOpenAIAPIVersion)
	for k, v := range AzureOpenAIModelMapper {
		log.Printf("loading azure model mapper: %s -> %s", k, v)
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

func handleToken(req *http.Request) {
	HandleToken(req)
}

func HandleToken(req *http.Request) {
	var token string

	// Check for API Key in the api-key header
	if apiKey := req.Header.Get("api-key"); apiKey != "" {
		token = apiKey
	} else if authHeader := req.Header.Get("Authorization"); authHeader != "" {
		// If not found, check for Authorization header
		token = strings.TrimPrefix(authHeader, "Bearer ")
	} else if AzureOpenAIToken != "" {
		// If neither is present, use the AzureOpenAIToken if set
		token = AzureOpenAIToken
	} else if envApiKey := os.Getenv("AZURE_OPENAI_API_KEY"); envApiKey != "" {
		// As a last resort, check for API key in environment variable
		token = envApiKey
	}

	if token != "" {
		// Set the api-key header with the found token
		req.Header.Set("api-key", token)
		// Remove the Authorization header to avoid conflicts
		req.Header.Del("Authorization")
	} else {
		log.Println("Warning: No authentication token found")
	}
}

// Update the makeDirector function to handle the new endpoint structure
func makeDirector(remote *url.URL) func(*http.Request) {
	return func(req *http.Request) {

		// Get model and map it to deployment
		model := getModelFromRequest(req)
		deployment := GetDeploymentByModel(model)

		// Handle token
		HandleToken(req)

		// Set the Host, Scheme, Path, and RawPath of the request
		originURL := req.URL.String()
		req.Host = remote.Host
		req.URL.Scheme = remote.Scheme
		req.URL.Host = remote.Host

		// Handle different endpoints
		switch {
		case strings.HasPrefix(req.URL.Path, "/v1/chat/completions"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "chat/completions")
		case strings.HasPrefix(req.URL.Path, "/v1/completions"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "completions")
		case strings.HasPrefix(req.URL.Path, "/v1/embeddings"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "embeddings")
		case strings.HasPrefix(req.URL.Path, "/v1/images/generations"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "images/generations")
		case strings.HasPrefix(req.URL.Path, "/v1/fine_tunes"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "fine-tunes")
		case strings.HasPrefix(req.URL.Path, "/v1/files"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "files")
		case strings.HasPrefix(req.URL.Path, "/v1/audio/speech"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "audio/speech")
		case strings.HasPrefix(req.URL.Path, "/v1/audio/transcriptions"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "transcriptions")
		case strings.HasPrefix(req.URL.Path, "/v1/audio/translations"):
			req.URL.Path = path.Join("/openai/deployments", deployment, "translations")
		default:
			req.URL.Path = path.Join("/openai/deployments", deployment, strings.TrimPrefix(req.URL.Path, "/v1/"))
		}

		req.URL.RawPath = req.URL.EscapedPath()

		// Add logging for new parameters
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

			// Restore the body to the request
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		}

		// Add the api-version query parameter
		query := req.URL.Query()
		query.Add("api-version", AzureOpenAIAPIVersion)
		req.URL.RawQuery = query.Encode()

		log.Printf("Proxying request [%s] %s -> %s", model, originURL, req.URL.String())
		log.Printf("Request Headers: %v", req.Header)
	}
}

func modifyResponse(res *http.Response) error {
	if res.StatusCode >= 400 {
		body, _ := io.ReadAll(res.Body)
		log.Printf("Azure API Error Response: Status: %d, Body: %s", res.StatusCode, string(body))
		res.Body = io.NopCloser(bytes.NewBuffer(body))
	}

	// Handle streaming responses
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
