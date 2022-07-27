
#include <CGUtils/math.hpp>
#include <CGUtils/model.hpp>
#include <CGUtils/image.hpp>
#include <CGUtils/file.hpp>
#include <SPIRV/GlslangToSpv.h>
#include "shaders.hpp"
#include "utils.hpp"
#include "camera.hpp"
using namespace wzz::math;
struct WindowDesc{
    vec2i res;
    std::string title;
    int msaa = 0;
};
static void errorCallback( int error, const char * description )
{
    fprintf( stderr, "GLFW Error %d: %s\n", error, description );
}
static void check_vk_result( VkResult err )
{
    if ( err != 0 )
    {
        std::cerr << "Vulkan error " << vk::to_string( static_cast<vk::Result>( err ) );
        if ( err < 0 )
        {
            abort();
        }
    }
}
class GLFWException:std::runtime_error{
public:
    GLFWException(const std::string& msg = "")
    : std::runtime_error("glfw error: " + msg)
    {}
};
class KullaContyIBLApp{
public:
    KullaContyIBLApp(const WindowDesc& desc){
        res = desc.res;
        msaa = desc.msaa;
        glfwSetErrorCallback( errorCallback );
        if ( !glfwInit() )
        {
            throw GLFWException("init failed");
        }
        if ( !glfwVulkanSupported() )
        {
            throw GLFWException("not support vulkan");
        }
        glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

        window = glfwCreateWindow(res.x,res.y,desc.title.c_str(),nullptr,nullptr);
        if(!window){
            throw GLFWException("failed to create window");
        }

        uint32_t                 glfwExtensionsCount;
        const char **            glfwExtensions = glfwGetRequiredInstanceExtensions( &glfwExtensionsCount );
        std::vector<std::string> instanceExtensions;
        instanceExtensions.reserve( glfwExtensionsCount + 1 );
        for ( uint32_t i = 0; i < glfwExtensionsCount; i++ )
        {
            instanceExtensions.push_back( glfwExtensions[i] );
        }
        instanceExtensions.push_back( VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME );

        instance = vk::su::createInstance( desc.title.c_str(), "Vulkan", {}, instanceExtensions );
#if !defined( NDEBUG )
        vk::DebugUtilsMessengerEXT debugUtilsMessenger = instance.createDebugUtilsMessengerEXT( vk::su::makeDebugUtilsMessengerCreateInfoEXT() );
#endif
        physical_device = instance.enumeratePhysicalDevices().front();

        auto ret = glfwCreateWindowSurface(static_cast<VkInstance>(instance),window,nullptr,reinterpret_cast<VkSurfaceKHR *>( &surface ));
        check_vk_result(ret);

        auto indexes = vk::su::findGraphicsAndPresentQueueFamilyIndex(physical_device,surface);
        if(indexes.first != indexes.second){
            throw std::runtime_error("low gpu not support graphics and present together...");
        }

        auto supportedFeatures = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>();
        device = vk::su::createDevice( physical_device,
                                                             indexes.first,
                                                             { VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                                               VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
                                                               VK_KHR_MAINTENANCE_3_EXTENSION_NAME,
                                                               VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                                                             &supportedFeatures.get<vk::PhysicalDeviceFeatures2>().features,
                                                             &supportedFeatures.get<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>() );

    }
    void frame(){

    }
    void run(){
        while(!glfwWindowShouldClose(window)){
            glfwPollEvents();
            processInput();

            frame();
        }
        device.waitIdle();
    }
    void processInput(){
        if(glfwGetKey(window,GLFW_KEY_Q) == GLFW_PRESS)
            glfwSetWindowShouldClose(window,1);
        if(glfwGetKey(window,GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            glfwSetInputMode(window,GLFW_CURSOR,(mouse_visible = !mouse_visible) ? GLFW_CURSOR_NORMAL:GLFW_CURSOR_DISABLED);
        fps_camera_t::UpdateParams update;
        if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS) update.front = true;

        if(mouse_visible)
            camera.update(update);
        camera.recalculate_matrics();
    }
private:
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::SurfaceKHR surface;
    GLFWwindow* window;
    vec2i res;
    int msaa;
    fps_camera_t camera;
    bool mouse_visible = true;

    struct{
        vk::DescriptorSetLayout sample_params_set_layout;
        vk::DescriptorSetLayout brdf_lut_set_layout;
        vk::DescriptorSetLayout diffuse_set_layout;
        vk::DescriptorSetLayout light_set_layout;

    };
};

int main(){
    try {
        KullaContyIBLApp({
            .res = {1280,720},
            .title = "KullaConty-IBL",
            .msaa = 4
        }).run();
    }
    catch (const std::exception & err) {
        std::cerr<< err.what() << std::endl;
    }
}