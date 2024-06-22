package main

import (
    "encoding/json"
    "fmt"
    "io"
    "github.com/gin-gonic/gin"
    "github.com/gyarbij/azure-oai-proxy/pkg/azure"
    "github.com/gyarbij/azure-oai-proxy/pkg/openai"
    "log"
    "net/http"
    "os"
    "time"
)

var (
    Address   = "0.0.0.0:11437"
    ProxyMode = "azure"
)

func init() {
    gin.SetMode(gin.ReleaseMode)
    if v := os.Getenv("AZURE_OPENAI_PROXY_ADDRESS"); v != "" {
        Address = v
    }
    if v := os.Getenv("AZURE_OPENAI_PROXY_MODE"); v != "" {
        ProxyMode = v
    }
    log.Printf("loading azure openai proxy address: %s", Address)
    log.Printf("loading azure openai proxy mode: %s", ProxyMode)
}

func main() {
    router := gin.Default()
    if ProxyMode == "azure" {
        router.GET("/v1/models", handleGetModels)
        router.OPTIONS("/v1/*path", handleOptions)
        // Existing routes
        router.POST("/v1/chat/completions", handleAzureProxy)
        router.POST("/v1/completions", handleAzureProxy)
        router.POST("/v1/embeddings", handleAzureProxy)
        // DALL-E routes
        router.POST("/v1/images/generations", handleAzureProxy)
        // speech- routes
        router.POST("/v1/audio/speech", handleAzureProxy)
        router.GET("/v1/audio/voices", handleAzureProxy)
        router.POST("/v1/audio/transcriptions", handleAzureProxy)
        router.POST("/v1/audio/translations", handleAzureProxy)
        // Fine-tuning routes
        router.POST("/v1/fine_tunes", handleAzureProxy)
        router.GET("/v1/fine_tunes", handleAzureProxy)
        router.GET("/v1/fine_tunes/:fine_tune_id", handleAzureProxy)
        router.POST("/v1/fine_tunes/:fine_tune_id/cancel", handleAzureProxy)
        router.GET("/v1/fine_tunes/:fine_tune_id/events", handleAzureProxy)
        // Files management routes
        router.POST("/v1/files", handleAzureProxy)
        router.GET("/v1/files", handleAzureProxy)
        router.DELETE("/v1/files/:file_id", handleAzureProxy)
        router.GET("/v1/files/:file_id", handleAzureProxy)
        router.GET("/v1/files/:file_id/content", handleAzureProxy)
        // Deployments management routes
        router.GET("/deployments", handleAzureProxy)
        router.GET("/deployments/:deployment_id", handleAzureProxy)
        router.GET("/v1/models/:model_id/capabilities", handleAzureProxy)
    } else {
        router.Any("*path", handleOpenAIProxy)
    }

    router.Run(Address)
}

func handleGetModels(c *gin.Context) {
    models, err := fetchDeployedModels()
    if err != nil {
        log.Printf("error fetching deployed models: %v", err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to fetch deployed models"})
        return
    }
    result := azure.ListModelResponse{
        Object: "list",
        Data:   models,
    }
    c.JSON(http.StatusOK, result)
}

func fetchDeployedModels() ([]azure.Model, error) {
    endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
    token := os.Getenv("AZURE_OPENAI_TOKEN")

    req, err := http.NewRequest("GET", endpoint+"/openai/deployments?api-version=2024-05-01-preview", nil)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Authorization", "Bearer "+token)

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("failed to fetch deployed models: %s", string(body))
    }

    var deployedModelsResponse azure.ListDeployedModelsResponse
    if err := json.NewDecoder(resp.Body).Decode(&deployedModelsResponse); err != nil {
        return nil, err
    }

    models := []azure.Model{}
    for _, deployedModel := range deployedModelsResponse.Data {
        createdTime, err := time.Parse(time.RFC3339, deployedModel.CreatedAt)
        if err != nil {
            log.Printf("Error parsing CreatedAt time: %v", err)
            continue
        }
        createdUnix := createdTime.Unix()

        models = append(models, azure.Model{
            ID:      deployedModel.ModelID,
            Object:  "model",
            Created: int(createdUnix),
            OwnedBy: "openai",
            Permission: []azure.ModelPermission{
                {
                    ID:                 "",
                    Object:             "model",
                    Created:            int(createdUnix),
                    AllowCreateEngine:  true,
                    AllowSampling:      true,
                    AllowLogprobs:      true,
                    AllowSearchIndices: true,
                    AllowView:          true,
                    AllowFineTuning:    true,
                    Organization:       "*",
                    Group:              nil,
                    IsBlocking:         false,
                },
            },
            Root:   deployedModel.ModelID,
            Parent: nil,
        })
    }

    return models, nil
}

func handleOptions(c *gin.Context) {
    c.Header("Access-Control-Allow-Origin", "*")
    c.Header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
    c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    c.Status(200)
    return
}

func handleAzureProxy(c *gin.Context) {
    if c.Request.Method == http.MethodOptions {
        handleOptions(c)
        return
    }

    server := azure.NewOpenAIReverseProxy()
    server.ServeHTTP(c.Writer, c.Request)

    if c.Writer.Header().Get("Content-Type") == "text/event-stream" {
        if _, err := c.Writer.Write([]byte("\n")); err != nil {
            log.Printf("rewrite azure response error: %v", err)
        }
    }

    // Enhanced error logging
    if c.Writer.Status() >= 400 {
        log.Printf("Azure API request failed: %s %s, Status: %d", c.Request.Method, c.Request.URL.Path, c.Writer.Status())
    }
}

func handleOpenAIProxy(c *gin.Context) {
    server := openai.NewOpenAIReverseProxy()
    server.ServeHTTP(c.Writer, c.Request)
}