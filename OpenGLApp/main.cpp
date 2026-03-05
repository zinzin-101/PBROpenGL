#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <filesystem.h>
#include <shader.h>
#include <camera.h>
//#include <model.h>
#include "PBRModel.h"

#include <iostream>
#include <map>

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
bool getKeyDown(GLFWwindow* window, unsigned int key);
unsigned int loadTexture(const char *path);
void renderSphere();
void renderCube();
void renderQuad();

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 2.0f));
float lastX = 800.0f / 2.0;
float lastY = 600.0 / 2.0;
bool firstMouse = true;
float normalMoveSpeed = 2.5f;
float fastMoveSpeed = 10.0f;

// timing
float deltaTime = 0.0f;	
float lastFrame = 0.0f;

// background rotation
float backgroundRotateAngle = 0.0f;
unsigned int canRotatebackgroundFlag = 0; // can rotate if first 2 bits are 1 | first bit = left alt, second bit = left click
float rotateSensitivity = 15.0f; // in degrees
bool toggleEnvironmentMap = true;

bool debugLightPosition = true;
bool debugShadowQuad = false;

// PBR material textures
// --------------------------
// rusted iron
unsigned int ironAlbedoMap;
unsigned int ironNormalMap;
unsigned int ironMetallicMap;
unsigned int ironRoughnessMap;
unsigned int ironAOMap;

// gold
unsigned int goldAlbedoMap;
unsigned int goldNormalMap;
unsigned int goldMetallicMap;
unsigned int goldRoughnessMap;
unsigned int goldAOMap;

// grass
unsigned int grassAlbedoMap;
unsigned int grassNormalMap;
unsigned int grassMetallicMap;
unsigned int grassRoughnessMap;
unsigned int grassAOMap;

// plastic
unsigned int plasticAlbedoMap;
unsigned int plasticNormalMap;
unsigned int plasticMetallicMap;
unsigned int plasticRoughnessMap;
unsigned int plasticAOMap;

// wall
unsigned int wallAlbedoMap;
unsigned int wallNormalMap;
unsigned int wallMetallicMap;
unsigned int wallRoughnessMap;
unsigned int wallAOMap;

void renderSceneDepth(Shader& shader, glm::vec3 oldLightPos);
int PBRMesh::maxTextureNumber = 0;

unsigned int PBRMesh::defaultAlbedo = 0;
unsigned int PBRMesh::defaultNormal = 0;
unsigned int PBRMesh::defaultMetallic = 0;
unsigned int PBRMesh::defaultRoughness = 0;
unsigned int PBRMesh::defaultAO = 0;

PBRModel* boxModelPtr = nullptr;
PBRModel* shotgunModelPtr = nullptr;
PBRModel* revolverGunModelPtr = nullptr;
PBRModel* groundModelPtr = nullptr;
PBRModel* chisaModelPtr = nullptr;
glm::mat4 boxModelMat = glm::mat4(1.0f);
glm::mat4 shotgunModelMat = glm::mat4(1.0f);
glm::mat4 revolverGunModelMat = glm::mat4(1.0f);
glm::mat4 groundModelMat = glm::mat4(1.0f);
glm::mat4 goldSphereModelMat = glm::mat4(1.0f);
glm::mat4 goldSphereModelMat2 = glm::mat4(1.0f);
glm::mat4 chisaModelMat = glm::mat4(1.0f);

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "PBROpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    // set depth function to less than AND equal for skybox depth trick.
    glDepthFunc(GL_LEQUAL);
    //glDepthFunc(GL_LESS);
    // enable seamless cubemap sampling for lower mip levels in the pre-filter map.
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // build and compile shaders
    // -------------------------
    Shader pbrShader("2.2.2.pbr.vs", "2.2.2.pbr.fs");
    Shader equirectangularToCubemapShader("2.2.2.cubemap.vs", "2.2.2.equirectangular_to_cubemap.fs");
    Shader irradianceShader("2.2.2.cubemap.vs", "2.2.2.irradiance_convolution.fs");
    Shader prefilterShader("2.2.2.cubemap.vs", "2.2.2.prefilter.fs");
    Shader brdfShader("2.2.2.brdf.vs", "2.2.2.brdf.fs");
    Shader backgroundShader("2.2.2.background.vs", "2.2.2.background.fs");

    Shader depthShader("3.1.3.shadow_mapping_depth.vs", "3.1.3.shadow_mapping_depth.fs");
    Shader debugDepthQuad("3.1.3.debug_quad.vs", "3.1.3.debug_quad_depth.fs");

    pbrShader.use();
    pbrShader.setInt("irradianceMap", 0);
    pbrShader.setInt("prefilterMap", 1);
    pbrShader.setInt("brdfLUT", 2);
    pbrShader.setInt("albedoMap1", 4);
    pbrShader.setInt("normalMap1", 5);
    pbrShader.setInt("metallicMap1", 6);
    pbrShader.setInt("roughnessMap1", 7);
    pbrShader.setInt("aoMap1", 8);
    
    // shadow map shader
    pbrShader.setInt("shadowMap", 3);
    debugDepthQuad.use();
    debugDepthQuad.setInt("depthMap", 3);

    backgroundShader.use();
    backgroundShader.setInt("environmentMap", 0);

    // load PBR material textures
    // --------------------------
    // rusted iron
    ironAlbedoMap = loadTexture(FileSystem::getPath("resources/textures/pbr/rusted_iron/albedo.png").c_str());
    ironNormalMap = loadTexture(FileSystem::getPath("resources/textures/pbr/rusted_iron/normal.png").c_str());
    ironMetallicMap = loadTexture(FileSystem::getPath("resources/textures/pbr/rusted_iron/metallic.png").c_str());
    ironRoughnessMap = loadTexture(FileSystem::getPath("resources/textures/pbr/rusted_iron/roughness.png").c_str());
    ironAOMap = loadTexture(FileSystem::getPath("resources/textures/pbr/rusted_iron/ao.png").c_str());

    // gold
    goldAlbedoMap = loadTexture(FileSystem::getPath("resources/textures/pbr/gold/albedo.png").c_str());
    goldNormalMap = loadTexture(FileSystem::getPath("resources/textures/pbr/gold/normal.png").c_str());
    goldMetallicMap = loadTexture(FileSystem::getPath("resources/textures/pbr/gold/metallic.png").c_str());
    goldRoughnessMap = loadTexture(FileSystem::getPath("resources/textures/pbr/gold/roughness.png").c_str());
    goldAOMap = loadTexture(FileSystem::getPath("resources/textures/pbr/gold/ao.png").c_str());

    // grass
    grassAlbedoMap = loadTexture(FileSystem::getPath("resources/textures/pbr/grass/albedo.png").c_str());
    grassNormalMap = loadTexture(FileSystem::getPath("resources/textures/pbr/grass/normal.png").c_str());
    grassMetallicMap = loadTexture(FileSystem::getPath("resources/textures/pbr/grass/metallic.png").c_str());
    grassRoughnessMap = loadTexture(FileSystem::getPath("resources/textures/pbr/grass/roughness.png").c_str());
    grassAOMap = loadTexture(FileSystem::getPath("resources/textures/pbr/grass/ao.png").c_str());

    // plastic
    plasticAlbedoMap = loadTexture(FileSystem::getPath("resources/textures/pbr/plastic/albedo.png").c_str());
    plasticNormalMap = loadTexture(FileSystem::getPath("resources/textures/pbr/plastic/normal.png").c_str());
    plasticMetallicMap = loadTexture(FileSystem::getPath("resources/textures/pbr/plastic/metallic.png").c_str());
    plasticRoughnessMap = loadTexture(FileSystem::getPath("resources/textures/pbr/plastic/roughness.png").c_str());
    plasticAOMap = loadTexture(FileSystem::getPath("resources/textures/pbr/plastic/ao.png").c_str());

    // wall
    wallAlbedoMap = loadTexture(FileSystem::getPath("resources/textures/pbr/wall/albedo.png").c_str());
    wallNormalMap = loadTexture(FileSystem::getPath("resources/textures/pbr/wall/normal.png").c_str());
    wallMetallicMap = loadTexture(FileSystem::getPath("resources/textures/pbr/wall/metallic.png").c_str());
    wallRoughnessMap = loadTexture(FileSystem::getPath("resources/textures/pbr/wall/roughness.png").c_str());
    wallAOMap = loadTexture(FileSystem::getPath("resources/textures/pbr/wall/ao.png").c_str());

    // lights
    // ------
    const int numOfLights = 4;
    glm::vec3 lightPositions[] = {
        //glm::vec3(12.5f,  2.5f, 10.0f), // the first one is used for directional light
        glm::vec3(5.0f,  2.5f, 4.0f),
        glm::vec3(5.0f,  2.5f, 4.0f),
        glm::vec3(5.0f,  2.5f, 4.0f),
        glm::vec3(5.0f,  2.5f, 4.0f),
        //glm::vec3(0.0f,  10.0f, 0.0f),
        //glm::vec3( 10.0f,  10.0f, 10.0f),
        //glm::vec3(-10.0f, -10.0f, 10.0f),
        //glm::vec3( 10.0f, -10.0f, 10.0f),
    };
    glm::vec3 lightColors[] = {
        glm::vec3(100.0f, 100.0f, 100.0f),
        glm::vec3(100.0f, 100.0f, 100.0f),
        glm::vec3(100.0f, 100.0f, 100.0f),
        glm::vec3(100.0f, 100.0f, 100.0f)
    };

    // pbr: setup framebuffer
    // ----------------------
    unsigned int captureFBO;
    unsigned int captureRBO;
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);

    // pbr: load the HDR environment map
    // ---------------------------------
    stbi_set_flip_vertically_on_load(true);
    int width, height, nrComponents;
    //float *data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/newport_loft.hdr").c_str(), &width, &height, &nrComponents, 0);
    //float* data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/morning_2k.hdr").c_str(), &width, &height, &nrComponents, 0);
    float* data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/puresky_2k.hdr").c_str(), &width, &height, &nrComponents, 0);
    //float* data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/studio.hdr").c_str(), &width, &height, &nrComponents, 0);
    //float* data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/pisztyk.hdr").c_str(), &width, &height, &nrComponents, 0);
    //float* data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/kloppenheim_puresky.hdr").c_str(), &width, &height, &nrComponents, 0);

    unsigned int hdrTexture;
    if (data)
    {
        glGenTextures(1, &hdrTexture);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data); // note how we specify the texture's data value to be float
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGB, GL_FLOAT, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Failed to load HDR image." << std::endl;
    }

    // pbr: setup cubemap to render to and attach to framebuffer
    // ---------------------------------------------------------
    unsigned int envCubemap;
    glGenTextures(1, &envCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // enable pre-filter mipmap sampling (combatting visible dots artifact)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions
    // ----------------------------------------------------------------------------------------------
    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[] =
    {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    // pbr: convert HDR equirectangular environment map to cubemap equivalent
    // ----------------------------------------------------------------------
    equirectangularToCubemapShader.use();
    equirectangularToCubemapShader.setInt("equirectangularMap", 0);
    equirectangularToCubemapShader.setMat4("projection", captureProjection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);

    glViewport(0, 0, 512, 512); // don't forget to configure the viewport to the capture dimensions.
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (unsigned int i = 0; i < 6; ++i)
    {
        equirectangularToCubemapShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderCube();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // then let OpenGL generate mipmaps from first mip face (combatting visible dots artifact)
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    // pbr: create an irradiance cubemap, and re-scale capture FBO to irradiance scale.
    // --------------------------------------------------------------------------------
    unsigned int irradianceMap;
    glGenTextures(1, &irradianceMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

    // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
    // -----------------------------------------------------------------------------
    irradianceShader.use();
    irradianceShader.setInt("environmentMap", 0);
    irradianceShader.setMat4("projection", captureProjection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

    glViewport(0, 0, 32, 32); // don't forget to configure the viewport to the capture dimensions.
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (unsigned int i = 0; i < 6; ++i)
    {
        irradianceShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderCube();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // pbr: create a pre-filter cubemap, and re-scale capture FBO to pre-filter scale.
    // --------------------------------------------------------------------------------
    unsigned int prefilterMap;
    glGenTextures(1, &prefilterMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // be sure to set minification filter to mip_linear 
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    // pbr: run a quasi monte-carlo simulation on the environment lighting to create a prefilter (cube)map.
    // ----------------------------------------------------------------------------------------------------
    prefilterShader.use();
    prefilterShader.setInt("environmentMap", 0);
    prefilterShader.setMat4("projection", captureProjection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    unsigned int maxMipLevels = 5;
    for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
    {
        // reisze framebuffer according to mip-level size.
        unsigned int mipWidth = static_cast<unsigned int>(128 * std::pow(0.5, mip));
        unsigned int mipHeight = static_cast<unsigned int>(128 * std::pow(0.5, mip));
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = (float)mip / (float)(maxMipLevels - 1);
        prefilterShader.setFloat("roughness", roughness);
        for (unsigned int i = 0; i < 6; ++i)
        {
            prefilterShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap, mip);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderCube();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // pbr: generate a 2D LUT from the BRDF equations used.
    // ----------------------------------------------------
    unsigned int brdfLUTTexture;
    glGenTextures(1, &brdfLUTTexture);

    // pre-allocate enough memory for the LUT texture.
    glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);
    // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // then re-configure capture framebuffer object and render screen-space quad with BRDF shader.
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

    glViewport(0, 0, 512, 512);
    brdfShader.use();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderQuad();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // shadow map: configure FBO
    // -------------------------
    const unsigned int SHADOW_WIDTH = 4096;
    const unsigned int SHADOW_HEIGHT = 4096;
    unsigned int depthMapFBO;
    glGenFramebuffers(1, &depthMapFBO);
    // create depth texture
    unsigned int depthMap;
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // initialize static shader uniforms before rendering
    // --------------------------------------------------
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    pbrShader.use();
    pbrShader.setMat4("projection", projection);
    backgroundShader.use();
    backgroundShader.setMat4("projection", projection);

    // then before rendering, configure the viewport to the original framebuffer's screen dimensions
    int scrWidth, scrHeight;
    glfwGetFramebufferSize(window, &scrWidth, &scrHeight);
    glViewport(0, 0, scrWidth, scrHeight);

    stbi_set_flip_vertically_on_load(false); // for loading model texture
    //PBRModel currentModel(FileSystem::getPath("resources/objects/wooden_chest/scene.gltf"));
    //PBRModel currentModel(FileSystem::getPath("resources/objects/chisa/scene.gltf"));
    PBRModel shotgunModel(FileSystem::getPath("resources/objects/shotgun/scene.gltf"));
    PBRModel revolverGunModel(FileSystem::getPath("resources/objects/revolvergun/scene.gltf"));
    PBRModel boxModel(FileSystem::getPath("resources/objects/military_box/scene.gltf"));
    //PBRModel currentModel(FileSystem::getPath("resources/objects/wheel/wheel.glb"));

    PBRModel groundModel(FileSystem::getPath("resources/objects/stone_ground/scene.gltf"));
    //PBRModel groundModel(FileSystem::getPath("resources/objects/rocky_ground/scene.gltf"));

    PBRModel chisaModel(FileSystem::getPath("resources/objects/chisa/scene.gltf"));

    shotgunModelPtr = &shotgunModel;
    revolverGunModelPtr = &revolverGunModel;
    boxModelPtr = &boxModel;
    groundModelPtr = &groundModel;
    chisaModelPtr = &chisaModel;

    // render loop
    // -----------
    std::cout << "entering render loop" << std::endl;
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // envmap rotation
        glm::mat4 envRotMat = glm::rotate(glm::mat4(1.0f), glm::radians(-backgroundRotateAngle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 invEnvRotMat = glm::rotate(glm::mat4(1.0f), glm::radians(backgroundRotateAngle), glm::vec3(0.0f, 1.0f, 0.0f));

        // update shader light positions
        for (unsigned int i = 0; i < 4; ++i)
        {
            //glm::vec3 newPos = lightPositions[i] + glm::vec3(sin(glfwGetTime() * 5.0) * 5.0, 0.0, 0.0);
            glm::vec3 newPos = glm::vec3(invEnvRotMat * glm::vec4(lightPositions[i], 1.0f));
            //newPos = lightPositions[i];
            pbrShader.setVec3("lightPositions[" + std::to_string(i) + "]", newPos);
        }

        // calculate model position
        boxModelMat = glm::mat4(1.0f);
        boxModelMat = glm::translate(boxModelMat, glm::vec3(0.0f, 0.0f, 0.0f));
        boxModelMat = glm::scale(boxModelMat, glm::vec3(0.01f));
        boxModelMat = glm::rotate(boxModelMat, glm::radians(-90.0f), glm::vec3(1, 0, 0));

        shotgunModelMat = glm::mat4(1.0f);
        shotgunModelMat = glm::translate(shotgunModelMat, glm::vec3(0.75f, 0.35f, 0.0f));
        shotgunModelMat = glm::scale(shotgunModelMat, glm::vec3(1.75f));
        shotgunModelMat = glm::rotate(shotgunModelMat, glm::radians(-90.0f), glm::vec3(0, 1, 0));
        shotgunModelMat = glm::rotate(shotgunModelMat, glm::radians(-76.0f), glm::vec3(1, 0, 0));

        revolverGunModelMat = glm::mat4(1.0f);
        revolverGunModelMat = glm::translate(revolverGunModelMat, glm::vec3(0.0f, 1.0f, 0.0f));
        revolverGunModelMat = glm::scale(revolverGunModelMat, glm::vec3(1.f));
        revolverGunModelMat = glm::rotate(revolverGunModelMat, glm::radians(-90.0f), glm::vec3(1, 0, 0));

        groundModelMat = glm::mat4(1.0f);
        groundModelMat = glm::translate(groundModelMat, glm::vec3(0.0f, -0.2, 0.0f));
        groundModelMat = glm::scale(groundModelMat, glm::vec3(2.0f, 0.5f, 2.0f));
        groundModelMat = glm::rotate(groundModelMat, glm::radians(-90.0f), glm::vec3(1, 0, 0));

        goldSphereModelMat = glm::mat4(1.0f);
        goldSphereModelMat = glm::translate(goldSphereModelMat, glm::vec3(1.5f, 1.0f, -1.0f));
        goldSphereModelMat2 = glm::scale(goldSphereModelMat, glm::vec3(0.25f));

        goldSphereModelMat2 = glm::mat4(1.0f);
        goldSphereModelMat2 = glm::translate(goldSphereModelMat2, glm::vec3(-1.5f, 1.0f, -1.0f));
        goldSphereModelMat2 = glm::scale(goldSphereModelMat2, glm::vec3(0.5f));

        chisaModelMat = glm::mat4(1.0f);
        chisaModelMat = glm::translate(chisaModelMat, glm::vec3(0.0f, -0.2f, -1.0f));
        chisaModelMat = glm::scale(chisaModelMat, glm::vec3(0.015f));
        chisaModelMat = glm::rotate(chisaModelMat, glm::radians(-90.0f), glm::vec3(1, 0, 0));


        // render scene depth to texture from light source
        // -----------------------------------------------
        glm::mat4 lightProjection;
        glm::mat4 lightView;
        glm::mat4 lightSpaceMatrix;
        float nearPlane = 2.0f;
        float farPlane = 15.0f;
        lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, nearPlane, farPlane);
        glm::vec3 lightPos = glm::vec3(invEnvRotMat * glm::vec4(lightPositions[0], 1.0f));
        lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        lightSpaceMatrix = lightProjection * lightView;
        depthShader.use();
        depthShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        renderSceneDepth(depthShader, lightPositions[0]);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // reset viewport
        glViewport(0, 0, scrWidth, scrHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // render scene, supplying the convoluted irradiance map to the final shader.
        // ------------------------------------------------------------------------------------------
        pbrShader.use();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = camera.GetViewMatrix();
        pbrShader.setMat4("view", view);
        pbrShader.setVec3("camPos", camera.Position);

        // bind pre-computed IBL data
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);

        // bind shadow map
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        
        pbrShader.setMat4("envMapRotation", envRotMat);
        pbrShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);

        pbrShader.setMat4("model", boxModelMat);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(boxModelMat))));
        boxModel.Draw(pbrShader);

        pbrShader.setMat4("model", shotgunModelMat);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(shotgunModelMat))));
        shotgunModel.Draw(pbrShader);

        pbrShader.setMat4("model", revolverGunModelMat);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(revolverGunModelMat))));
        revolverGunModel.Draw(pbrShader);

        pbrShader.setMat4("model", chisaModelMat);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(chisaModelMat))));
        chisaModel.Draw(pbrShader);

        pbrShader.setMat4("model", groundModelMat);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(groundModelMat))));
        groundModel.Draw(pbrShader);


        // gold
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, goldAlbedoMap);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, goldNormalMap);
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, goldMetallicMap);
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, goldRoughnessMap);
        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_2D, goldAOMap);
        pbrShader.setMat4("model", goldSphereModelMat);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(goldSphereModelMat))));
        renderSphere();

        pbrShader.setMat4("model", goldSphereModelMat2);
        pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(goldSphereModelMat2))));
        renderSphere();

        //model = glm::mat4(1.0f);
        ////model = glm::translate(model, glm::vec3(-3.0, 0.0, 2.0));
        //glm::vec3 pos = lightPositions[0] * 0.5f;
        //model = glm::translate(model, pos);
        //model = glm::scale(model, glm::vec3(2.5f));
        //pbrShader.setMat4("model", model);
        //pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));
        //renderSphere();

        // render light source (simply re-render sphere at light positions)
        // this looks a bit off as we use the same shader, but it'll make their positions obvious and 
        // keeps the codeprint small.
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, plasticAlbedoMap);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, plasticNormalMap);
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, plasticMetallicMap);
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, plasticRoughnessMap);
        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_2D, plasticAOMap);
        for (unsigned int i = 0; i < numOfLights; ++i)
        {
            //glm::vec3 newPos = lightPositions[i] + glm::vec3(sin(glfwGetTime() * 5.0) * 5.0, 0.0, 0.0);
            glm::vec3 newPos = glm::vec3(invEnvRotMat * glm::vec4(lightPositions[i], 1.0f));
            //newPos = lightPositions[i];
            pbrShader.setVec3("lightPositions[" + std::to_string(i) + "]", newPos);
            pbrShader.setVec3("lightColors[" + std::to_string(i) + "]", lightColors[i]);

            model = glm::mat4(1.0f);
            model = glm::translate(model, newPos);
            model = glm::scale(model, glm::vec3(0.5f));
            pbrShader.setMat4("model", model);
            pbrShader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

            if (debugLightPosition) renderSphere();
        }

        // render skybox (render as last to prevent overdraw)
        backgroundShader.use();

        backgroundShader.setMat4("view", view);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
        //glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap); // display irradiance map
        //glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap); // display prefilter map

        model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(backgroundRotateAngle), glm::vec3(0.0f, 1.0f, 0.0f));
        backgroundShader.setMat4("model", model);
        if (toggleEnvironmentMap) renderCube();

        // shadow map debug quad
        debugDepthQuad.use();
        debugDepthQuad.setFloat("near_plane", nearPlane);
        debugDepthQuad.setFloat("far_plane", farPlane);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        if (debugShadowQuad) renderQuad();

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        //std::cout << "max texture number: " << PBRMesh::maxTextureNumber << std::endl;
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

void renderSceneDepth(Shader& shader, glm::vec3 oldLightPos) {
    shader.setMat4("model", boxModelMat);
    boxModelPtr->Draw(shader);

    shader.setMat4("model", shotgunModelMat);
    shotgunModelPtr->Draw(shader);

    shader.setMat4("model", revolverGunModelMat);
    revolverGunModelPtr->Draw(shader);

    shader.setMat4("model", groundModelMat);
    groundModelPtr->Draw(shader);

    shader.setMat4("model", goldSphereModelMat);
    renderSphere();

    shader.setMat4("model", goldSphereModelMat2);
    renderSphere();

    shader.setMat4("model", chisaModelMat);
    chisaModelPtr->Draw(shader);

    //model = glm::mat4(1.0f);
    ////model = glm::translate(model, glm::vec3(-3.0, 0.0, 2.0));
    //glm::vec3 pos = oldLightPos * 0.5f;
    //model = glm::translate(model, pos);
    //model = glm::scale(model, glm::vec3(2.5f));
    //shader.setMat4("model", model);
    //renderSphere();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
std::map<unsigned int, bool> keyDownMap;
bool getKeyDown(GLFWwindow* window, unsigned int key) {
    // init
    if (!keyDownMap.contains(key)) {
        keyDownMap[key] = false;
        return false;
    }

    if (glfwGetKey(window, key) == GLFW_PRESS && keyDownMap.at(key)) {
        return false;
    }

    if (glfwGetKey(window, key) == GLFW_PRESS && !keyDownMap.at(key)) {
        keyDownMap[key] = true;
        return true;
    }

    if (glfwGetKey(window, key) == GLFW_RELEASE && keyDownMap.at(key)) {
        keyDownMap[key] = false;
        return false;
    }

    return false;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    unsigned int leftAltDownStatus = glfwGetKey(window, GLFW_KEY_LEFT_ALT);
    if (leftAltDownStatus == GLFW_PRESS) {
        canRotatebackgroundFlag |= 1; // first bit 01
    }
    else {
        canRotatebackgroundFlag &= ~1;
    }

    if (getKeyDown(window, GLFW_KEY_B)) {
        toggleEnvironmentMap = !toggleEnvironmentMap;
    }

    if (canRotatebackgroundFlag != 3) {
        glm::vec3 movement(0.0f);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            movement += camera.Front;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            movement -= camera.Front;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            movement -= camera.Right;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            movement += camera.Right;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            movement += camera.Up;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            movement -= camera.Up;

        movement = glm::normalize(movement);
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            movement *= fastMoveSpeed;
        }
        else {
            movement *= normalMoveSpeed;
        }

        if (glm::length(movement) > 0.5f) {
            camera.Position += movement * deltaTime;
        }
    }


    //if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
    //    backgroundRotateAngle += 30.0f * deltaTime;
    //    std::cout << "env rotation angle: " << backgroundRotateAngle << std::endl;
    //}
    //if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
    //    backgroundRotateAngle -= 30.0f * deltaTime;
    //    std::cout << "env rotation angle: " << backgroundRotateAngle << std::endl;
    //}
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouseCallback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    bool isRotatingEnvMap = canRotatebackgroundFlag == 3; // first 2 bits are 1: 11

    if (isRotatingEnvMap) {
        backgroundRotateAngle += xoffset * rotateSensitivity * deltaTime;
    }
    else {
        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        canRotatebackgroundFlag |= 2; // second bit 10
    }
    else {
        canRotatebackgroundFlag &= ~2;
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
 
// renders (and builds at first invocation) a sphere
// -------------------------------------------------
unsigned int sphereVAO = 0;
GLsizei indexCount;
void renderSphere()
{
    if (sphereVAO == 0)
    {
        glGenVertexArrays(1, &sphereVAO);

        unsigned int vbo, ebo;
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        std::vector<glm::vec3> positions;
        std::vector<glm::vec2> uv;
        std::vector<glm::vec3> normals;
        std::vector<unsigned int> indices;

        const unsigned int X_SEGMENTS = 64;
        const unsigned int Y_SEGMENTS = 64;
        const float PI = 3.14159265359f;
        for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
        {
            for (unsigned int y = 0; y <= Y_SEGMENTS; ++y)
            {
                float xSegment = (float)x / (float)X_SEGMENTS;
                float ySegment = (float)y / (float)Y_SEGMENTS;
                float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
                float yPos = std::cos(ySegment * PI);
                float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

                positions.push_back(glm::vec3(xPos, yPos, zPos));
                uv.push_back(glm::vec2(xSegment, ySegment));
                normals.push_back(glm::vec3(xPos, yPos, zPos));
            }
        }

        bool oddRow = false;
        for (unsigned int y = 0; y < Y_SEGMENTS; ++y)
        {
            if (!oddRow) // even rows: y == 0, y == 2; and so on
            {
                for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
                {
                    indices.push_back(y * (X_SEGMENTS + 1) + x);
                    indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                }
            }
            else
            {
                for (int x = X_SEGMENTS; x >= 0; --x)
                {
                    indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                    indices.push_back(y * (X_SEGMENTS + 1) + x);
                }
            }
            oddRow = !oddRow;
        }
        indexCount = static_cast<GLsizei>(indices.size());

        std::vector<float> data;
        for (unsigned int i = 0; i < positions.size(); ++i)
        {
            data.push_back(positions[i].x);
            data.push_back(positions[i].y);
            data.push_back(positions[i].z);
            if (normals.size() > 0)
            {
                data.push_back(normals[i].x);
                data.push_back(normals[i].y);
                data.push_back(normals[i].z);
            }
            if (uv.size() > 0)
            {
                data.push_back(uv[i].x);
                data.push_back(uv[i].y);
            }
        }
        glBindVertexArray(sphereVAO);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
        unsigned int stride = (3 + 2 + 3) * sizeof(float);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    }

    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);
}

// renderCube() renders a 1x1 3D cube in NDC.
// -------------------------------------------------
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
void renderCube()
{
    // initialize (if necessary)
    if (cubeVAO == 0)
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
            // front face
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            // left face
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            // right face
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
            // bottom face
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            // top face
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
             1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
             1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
        };
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        // fill buffer
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // link vertex attributes
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    // render Cube
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// utility function for loading a 2D texture from file
// ---------------------------------------------------
unsigned int loadTexture(char const * path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}
