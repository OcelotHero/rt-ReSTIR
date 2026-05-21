extern crate winit;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

use ash::{ext, khr, vk};

use std::ffi::{CStr, CString};
use std::mem::{offset_of, size_of};
use std::os::raw::c_void;
use std::sync::Arc;

use cgmath::{Deg, Matrix4, Point3, Vector3};

use anyhow::{Context, Result, bail, ensure};

use inline_spirv::include_spirv;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct ValidationInfo {
    enabled: bool,
    required_validation_layers: [&'static CStr; 1],
}
const VALIDATION: ValidationInfo =
    ValidationInfo { enabled: true, required_validation_layers: [c"VK_LAYER_KHRONOS_validation"] };

#[cfg(not(target_os = "macos"))]
const DEVICE_EXTENSIONS: [&'static CStr; 3] =
    [c"VK_KHR_swapchain", c"VK_KHR_dynamic_rendering", c"VK_KHR_synchronization2"];
#[cfg(target_os = "macos")]
const DEVICE_EXTENSIONS: [&'static CStr; 4] = [
    c"VK_KHR_swapchain",
    c"VK_KHR_portability_subset",
    c"VK_KHR_dynamic_rendering",
    c"VK_KHR_synchronization2",
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Self, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, color) as u32),
        ]
    }
}

const VERTICES: [Vertex; 4] = [
    Vertex { pos: [-0.5, -0.5], color: [1.0, 0.0, 0.0] },
    Vertex { pos: [0.5, -0.5], color: [0.0, 1.0, 0.0] },
    Vertex { pos: [0.5, 0.5], color: [0.0, 0.0, 1.0] },
    Vertex { pos: [-0.5, 0.5], color: [1.0, 1.0, 1.0] },
];

const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

#[derive(Default, Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn find_queue_family(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<QueueFamilyIndices> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        let mut queue_family_indices = QueueFamilyIndices::default();

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            let present_support = unsafe {
                surface_loader.get_physical_device_surface_support(*device, index as u32, *surface)
            };
            if queue_family.queue_count > 0 && present_support.unwrap_or(false) {
                queue_family_indices.present_family = Some(index as u32);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        Ok(queue_family_indices)
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }
        .to_str()
        .expect("Failed to convert debug message.")
        .to_owned();
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

fn populate_debug_messenger_create_info(create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT) {
    create_info.message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    create_info.message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;
    create_info.pfn_user_callback = Some(debug_callback);
}

struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanContext {
    pub fn init(window: &Window, required_extensions: &[*const i8]) -> Result<Self> {
        let entry = ash::Entry::linked();
        let instance = Self::create_instance(&entry, required_extensions)?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let mut context = Self {
            entry,
            instance: instance,
            surface_loader,
            surface: vk::SurfaceKHR::null(),
            debug_utils: None,
        };
        context.surface = Self::create_surface(&context.entry, &context.instance, window)?;
        context.debug_utils = Self::setup_debug_messenger(&context.entry, &context.instance)?;

        Ok(context)
    }

    fn create_instance(
        entry: &ash::Entry,
        required_extensions: &[*const i8],
    ) -> Result<ash::Instance> {
        ensure!(
            !VALIDATION.enabled || Self::check_validation_layer_support(entry)?,
            "Validation layers requested, but not available!"
        );

        let (major, minor) = match unsafe { entry.try_enumerate_instance_version()? } {
            // Vulkan 1.1+
            Some(version) => (vk::api_version_major(version), vk::api_version_minor(version)),
            // Vulkan 1.0
            None => (1, 0),
        };
        println!("Vulkan {}.{} supported", major, minor);

        let app_name = CString::new("Hello Triangle")?;
        let engine_name = CString::new("No Engine")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(
                0,
                env!("CARGO_PKG_VERSION_MAJOR").parse()?,
                env!("CARGO_PKG_VERSION_MINOR").parse()?,
                env!("CARGO_PKG_VERSION_PATCH").parse()?,
            ))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, major, minor, 0));

        let mut required_extensions = required_extensions.to_vec();
        #[cfg(target_os = "macos")]
        {
            required_extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());
            required_extensions.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }
        if VALIDATION.enabled {
            required_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        }

        let validation_layers: Vec<*const i8> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);

        #[cfg(target_os = "macos")]
        {
            create_info = create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
        populate_debug_messenger_create_info(&mut debug_create_info);
        if VALIDATION.enabled {
            create_info = create_info
                .enabled_layer_names(&validation_layers)
                .push_next(&mut debug_create_info);
        }

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .context("Failed to create Vulkan instance")?
        };

        Ok(instance)
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> Result<bool> {
        let layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .context("Failed to enumerate Instance Layers Properties!")?
        };

        if layer_properties.is_empty() {
            eprintln!("No available layers.");
            return Ok(false);
        } else {
            println!("Instance Available Layers: ");
            for layer in layer_properties.iter() {
                let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                println!("\t{:?}", layer_name);
            }
        }

        'layer: for layer_name in VALIDATION.required_validation_layers.iter() {
            for layer_property in layer_properties.iter() {
                let test_layer_name =
                    unsafe { CStr::from_ptr(layer_property.layer_name.as_ptr()) };
                if (*layer_name) == test_layer_name {
                    let api_version = layer_property.spec_version;
                    let major = vk::api_version_major(api_version);
                    let minor = vk::api_version_minor(api_version);
                    let patch = vk::api_version_patch(api_version);
                    println!(
                        "Validation layer {:?} is supported with version {}.{}.{}",
                        layer_name, major, minor, patch
                    );
                    continue 'layer;
                }
            }
            return Ok(false);
        }

        Ok(true)
    }

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>> {
        if !VALIDATION.enabled {
            return Ok(None);
        }

        let mut create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
        populate_debug_messenger_create_info(&mut create_info);

        let loader = ext::debug_utils::Instance::new(entry, instance);
        let messenger = unsafe {
            loader
                .create_debug_utils_messenger(&create_info, None)
                .context("Failed to create Debug Utils Messenger!")?
        };

        Ok(Some((loader, messenger)))
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .context("Failed to create Surface!")?
        };

        Ok(surface)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            if let Some((loader, messenger)) = self.debug_utils.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

struct DeviceContext {
    vk_ctx: Arc<VulkanContext>,
    dyn_render: khr::dynamic_rendering::Device,
    sync2: khr::synchronization2::Device,
    handle: ash::Device,
    device: vk::PhysicalDevice,
    indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
}

impl DeviceContext {
    pub fn init(vk_ctx: Arc<VulkanContext>) -> Result<Self> {
        let instance = &vk_ctx.instance;
        let surface_loader = &vk_ctx.surface_loader;
        let surface = &vk_ctx.surface;

        let (indices, device) = Self::pick_physical_device(instance, surface_loader, surface)?;
        let handle = Self::create_logical_device(instance, &device, &indices)?;
        let mut device = Self {
            vk_ctx: vk_ctx.clone(),
            device,
            indices,
            dyn_render: khr::dynamic_rendering::Device::new(instance, &handle),
            sync2: khr::synchronization2::Device::new(instance, &handle),
            handle,
            graphics_queue: vk::Queue::null(),
            present_queue: vk::Queue::null(),
        };

        device.graphics_queue =
            unsafe { device.handle.get_device_queue(indices.graphics_family.unwrap(), 0) };
        device.present_queue =
            unsafe { device.handle.get_device_queue(indices.present_family.unwrap(), 0) };

        Ok(device)
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(QueueFamilyIndices, vk::PhysicalDevice)> {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .context("Failed to enumerate Physical Devices!")?
        };

        println!("{} devices (GPU) found with vulkan support.", physical_devices.len());

        let (indices, device) = 'selection: {
            for device in physical_devices.iter() {
                if let (indices, true) =
                    Self::is_device_suitable(instance, device, surface_loader, surface)?
                {
                    break 'selection (indices, *device);
                }
            }
            bail!("Failed to find a suitable GPU!")
        };

        Ok((indices, device))
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(QueueFamilyIndices, bool)> {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };
        let device_name = unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) };
        let api_version = device_properties.api_version;
        let major = vk::api_version_major(api_version);
        let minor = vk::api_version_minor(api_version);
        let patch = vk::api_version_patch(api_version);
        println!("\tDevice Name: {:?}, Version: {}.{}.{}", device_name, major, minor, patch);

        let indices =
            QueueFamilyIndices::find_queue_family(instance, device, surface_loader, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, device)?;
        let swapchain_adequate = extensions_supported && {
            let (_, formats, present_modes) =
                SwapchainBundle::query_swapchain_support(device, surface_loader, surface)?;
            !formats.is_empty() && !present_modes.is_empty()
        };

        let mut dyn_render_features = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default();
        let mut sync2_features = vk::PhysicalDeviceSynchronization2FeaturesKHR::default();
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features13)
            .push_next(&mut dyn_render_features)
            .push_next(&mut sync2_features);
        unsafe { instance.get_physical_device_features2(*device, &mut features2) };

        let has_dyn_render = features13.dynamic_rendering == vk::TRUE
            || dyn_render_features.dynamic_rendering == vk::TRUE;
        let has_sync2 =
            features13.synchronization2 == vk::TRUE || sync2_features.synchronization2 == vk::TRUE;
        let dyn_render_supported = has_dyn_render && has_sync2;

        Ok((
            indices,
            indices.is_complete()
                && extensions_supported
                && swapchain_adequate
                && dyn_render_supported,
        ))
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
    ) -> Result<bool> {
        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .context("Failed to enumerate Device Extension Properties!")?
        };

        if extension_properties.is_empty() {
            eprintln!("No available device extensions.");
            return Ok(false);
        } else {
            println!("Available Device Extensions: ");
            for extension in extension_properties.iter() {
                let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                println!("\t\tName: {:?}, Version: {}", extension_name, extension.spec_version);
            }
        }

        'extension: for required_extension in DEVICE_EXTENSIONS.iter() {
            for extension in extension_properties.iter() {
                let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                if (*required_extension) == extension_name {
                    continue 'extension;
                }
            }
            return Ok(false);
        }

        Ok(true)
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        indices: &QueueFamilyIndices,
    ) -> Result<ash::Device> {
        let unique_queue_families = [indices.graphics_family, indices.present_family]
            .iter()
            .filter_map(|&family| family)
            .collect::<std::collections::HashSet<u32>>();

        let queue_priority = [1.0_f32];
        let queue_create_infos = unique_queue_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priority)
            })
            .collect::<Vec<_>>();

        let device_extensions: Vec<*const i8> =
            DEVICE_EXTENSIONS.iter().map(|ext_name| ext_name.as_ptr()).collect();

        let mut dyn_render_feature =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut sync2_feature =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut dyn_render_feature)
            .push_next(&mut sync2_feature)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions);

        let device = unsafe {
            instance
                .create_device(*physical_device, &create_info, None)
                .context("Failed to create Logical Device!")?
        };

        Ok(device)
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.handle.device_wait_idle().ok();
            self.handle.destroy_device(None);
        }
    }
}

struct Defer<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Defer<F> {
    pub fn new(f: F) -> Self {
        Self(Some(f))
    }
}

impl<F: FnOnce()> Drop for Defer<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

struct PipelineBundle {
    dev_ctx: Arc<DeviceContext>,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl PipelineBundle {
    pub fn new(dev_ctx: Arc<DeviceContext>, ubo_layout: &vk::DescriptorSetLayout) -> Result<Self> {
        let handle = &dev_ctx.handle;
        let device = &dev_ctx.device;
        let surface_loader = &dev_ctx.vk_ctx.surface_loader;
        let surface = &dev_ctx.vk_ctx.surface;

        let (_, formats, _) =
            SwapchainBundle::query_swapchain_support(device, surface_loader, surface)?;
        let format = SwapchainBundle::choose_swapchain_format(&formats).format;

        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        bundle.layout = Self::create_pipeline_layout(handle, ubo_layout)?;
        bundle.pipeline = Self::create_graphics_pipeline(handle, &bundle.layout, &format)?;

        Ok(bundle)
    }

    fn create_pipeline_layout(
        device: &ash::Device,
        ubo_layout: &vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout> {
        let ubo_layout = [*ubo_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&ubo_layout);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create Pipeline Layout!")?
        };

        Ok(pipeline_layout)
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        pipeline_layout: &vk::PipelineLayout,
        format: &vk::Format,
    ) -> Result<vk::Pipeline> {
        let vert_code = include_spirv!("shaders/shader.vert", vert);
        let frag_code = include_spirv!("shaders/shader.frag", frag);

        let vert_module = Self::create_shader_module(device, vert_code)?;
        let _vert_defer =
            Defer::new(|| unsafe { device.destroy_shader_module(vert_module, None) });
        let frag_module = Self::create_shader_module(device, frag_code)?;
        let _frag_defer =
            Defer::new(|| unsafe { device.destroy_shader_module(frag_module, None) });

        let main_function_name = CString::new("main")?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main_function_name),
        ];

        let binding_description = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_description)
            .vertex_attribute_descriptions(&attribute_descriptions);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state =
            vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment);

        let formats = [*format];
        let mut rendering =
            vk::PipelineRenderingCreateInfoKHR::default().color_attachment_formats(&formats);

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(*pipeline_layout)
            .push_next(&mut rendering)];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .map_err(|(_, err)| err)
                .context("Failed to create Graphics Pipeline!")?
        };

        Ok(graphics_pipelines[0])
    }

    fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);

        let shader_module = unsafe {
            device
                .create_shader_module(&create_info, None)
                .context("Failed to create Shader Module!")?
        };

        Ok(shader_module)
    }
}

impl Drop for PipelineBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_pipeline(self.pipeline, None);
            self.dev_ctx.handle.destroy_pipeline_layout(self.layout, None);
        }
    }
}

struct SwapchainBundle {
    dev_ctx: Arc<DeviceContext>,
    loader: khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

impl SwapchainBundle {
    fn new(dev_ctx: Arc<DeviceContext>) -> Result<Self> {
        let physical_device = &dev_ctx.device;
        let device = &dev_ctx.handle;
        let indices = &dev_ctx.indices;
        let instance = &dev_ctx.vk_ctx.instance;
        let surface_loader = &dev_ctx.vk_ctx.surface_loader;
        let surface = &dev_ctx.vk_ctx.surface;

        let loader = khr::swapchain::Device::new(instance, device);

        let (handle, format, extent, images) =
            Self::create_swapchain(physical_device, surface_loader, surface, indices, &loader)?;

        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            loader,
            handle,
            extent,
            images,
            image_views: Vec::new(),
        };
        bundle.image_views = Self::create_image_views(&device, &format, &bundle.images)?;

        Ok(bundle)
    }

    pub fn query_swapchain_support(
        physical_device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(vk::SurfaceCapabilitiesKHR, Vec<vk::SurfaceFormatKHR>, Vec<vk::PresentModeKHR>)>
    {
        unsafe {
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .context("Failed to query Surface Capabilities!")?;
            let formats = surface_loader
                .get_physical_device_surface_formats(*physical_device, *surface)
                .context("Failed to query Surface Formats!")?;
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .context("Failed to query Surface Present Modes!")?;

            Ok((capabilities, formats, present_modes))
        }
    }

    fn create_swapchain(
        physical_device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
        indices: &QueueFamilyIndices,
        loader: &khr::swapchain::Device,
    ) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
        let (capabilities, formats, present_modes) =
            Self::query_swapchain_support(physical_device, surface_loader, surface)?;

        let format = Self::choose_swapchain_format(&formats);
        let mode = Self::choose_swapchain_present_mode(&present_modes);
        let extent = Self::choose_swapchain_extent(&capabilities);

        let image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count > 0 {
            image_count.min(capabilities.max_image_count)
        } else {
            image_count
        };

        let families = [indices.graphics_family.unwrap(), indices.present_family.unwrap()];

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true);

        let create_info = match indices.graphics_family == indices.present_family {
            true => create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE),
            false => create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families),
        };

        let swapchain = unsafe {
            loader.create_swapchain(&create_info, None).context("Failed to create Swapchain!")?
        };
        let mut cleanup = Defer::new(|| unsafe { loader.destroy_swapchain(swapchain, None) });

        let swapchain_images = unsafe {
            loader.get_swapchain_images(swapchain).context("Failed to get Swapchain Images!")?
        };
        cleanup.0.take();

        Ok((swapchain, format.format, extent, swapchain_images))
    }

    pub fn choose_swapchain_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
            .clone()
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        *available_present_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO)
    }

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            use num::clamp;

            vk::Extent2D {
                width: clamp(
                    WIDTH,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    HEIGHT,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_image_views(
        device: &ash::Device,
        format: &vk::Format,
        images: &Vec<vk::Image>,
    ) -> Result<Vec<vk::ImageView>> {
        let mut image_views = Vec::with_capacity(images.len());

        for &image in images.iter() {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(*format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_view = unsafe {
                device
                    .create_image_view(&create_info, None)
                    .context("Failed to create Image View!")?
            };
            image_views.push(image_view);
        }

        Ok(image_views)
    }
}

impl Drop for SwapchainBundle {
    fn drop(&mut self) {
        unsafe {
            for &image_view in self.image_views.iter() {
                self.dev_ctx.handle.destroy_image_view(image_view, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

struct CommandBundle {
    dev_ctx: Arc<DeviceContext>,
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

impl CommandBundle {
    fn new(dev_ctx: Arc<DeviceContext>, max_frames: usize) -> Result<Self> {
        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            pool: Self::create_command_pool(
                &dev_ctx.handle,
                dev_ctx.indices.graphics_family.unwrap(),
            )?,
            buffers: Vec::new(),
        };
        bundle.buffers =
            Self::create_command_buffers(&bundle.dev_ctx.handle, &bundle.pool, max_frames)?;

        Ok(bundle)
    }

    fn create_command_pool(device: &ash::Device, index: u32) -> Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("Failed to create Command Pool!")?
        };

        Ok(command_pool)
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        size: usize,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(size as u32);

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .context("Failed to allocate Command Buffers!")?
        };

        Ok(command_buffers)
    }

    pub fn copy_buffer(
        &self,
        queue: vk::Queue,
        src: &BufferBundle,
        dst: &BufferBundle,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let device = &self.dev_ctx.handle;

        let cmd = [Self::create_command_buffers(device, &self.pool, 1)?[0]];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(cmd[0], &begin_info)
                .context("Failed to begin recording Command Buffer!")?;

            let copy_region = vk::BufferCopy::default().size(size);
            device.cmd_copy_buffer(cmd[0], src.buffer, dst.buffer, &[copy_region]);

            device
                .end_command_buffer(cmd[0])
                .context("Failed to end recording Command Buffer!")?;

            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd);
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;

            // Block CPU until execution finishes
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(self.pool, &cmd);
        }

        Ok(())
    }
}

impl Drop for CommandBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_command_pool(self.pool, None);
        }
    }
}

struct BufferBundle {
    dev_ctx: Arc<DeviceContext>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl BufferBundle {
    pub fn new(
        dev_ctx: Arc<DeviceContext>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            buffer: unsafe { dev_ctx.handle.create_buffer(&buffer_info, None)? },
            memory: vk::DeviceMemory::null(),
            size,
        };

        let mem_requirements =
            unsafe { dev_ctx.handle.get_buffer_memory_requirements(bundle.buffer) };
        let mem_type = Self::find_memory_type(
            &dev_ctx.vk_ctx.instance,
            &dev_ctx.device,
            mem_requirements.memory_type_bits,
            &properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        bundle.memory = unsafe { dev_ctx.handle.allocate_memory(&alloc_info, None)? };

        unsafe { dev_ctx.handle.bind_buffer_memory(bundle.buffer, bundle.memory, 0)? };

        Ok(bundle)
    }

    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        type_filter: u32,
        properties: &vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };

        for (i, mem_type) in mem_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0 && mem_type.property_flags.contains(*properties) {
                return Ok(i as u32);
            }
        }

        bail!("Failed to find suitable memory type!")
    }

    pub fn fill_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        let size = (size_of::<T>() * data.len()) as vk::DeviceSize;
        ensure!(size <= self.size, "Data size exceeds buffer capacity!");

        unsafe {
            let data_ptr = self.dev_ctx.handle.map_memory(
                self.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;

            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr as *mut T, data.len());

            self.dev_ctx.handle.unmap_memory(self.memory);
        }
        Ok(())
    }
}

impl Drop for BufferBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_buffer(self.buffer, None);
            self.dev_ctx.handle.free_memory(self.memory, None);
        }
    }
}

struct GeometryBundle {
    v_buffer: BufferBundle,
    i_buffer: BufferBundle,
}

impl GeometryBundle {
    pub fn new<V: Copy, I: Copy>(
        dev_ctx: Arc<DeviceContext>,
        command: &CommandBundle,
        vertices: &[V],
        indices: &[I],
    ) -> Result<Self> {
        let v_size = std::mem::size_of_val(vertices) as vk::DeviceSize;
        let i_size = std::mem::size_of_val(indices) as vk::DeviceSize;

        let v_buffer = {
            let staging = BufferBundle::new(
                dev_ctx.clone(),
                v_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            staging.fill_buffer(vertices)?;

            let dest = BufferBundle::new(
                dev_ctx.clone(),
                v_size,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            CommandBundle::copy_buffer(command, dev_ctx.graphics_queue, &staging, &dest, v_size)?;
            dest
        };

        let i_buffer = {
            let staging = BufferBundle::new(
                dev_ctx.clone(),
                i_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            staging.fill_buffer(indices)?;

            let dest = BufferBundle::new(
                dev_ctx.clone(),
                i_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            CommandBundle::copy_buffer(command, dev_ctx.graphics_queue, &staging, &dest, i_size)?;
            dest
        };

        Ok(Self { v_buffer, i_buffer })
    }
}

struct UniformBundle {
    dev_ctx: Arc<DeviceContext>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    buffers: Vec<BufferBundle>,
    mapped_ptrs: Vec<*mut c_void>,
    sets: Vec<vk::DescriptorSet>,
}

impl UniformBundle {
    pub fn new(dev_ctx: Arc<DeviceContext>, size: usize) -> Result<Self> {
        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            layout: Self::create_ubo_layout(&dev_ctx.handle)?,
            pool: vk::DescriptorPool::null(),
            buffers: Vec::new(),
            sets: Vec::new(),
            mapped_ptrs: Vec::new(),
        };
        (bundle.buffers, bundle.mapped_ptrs) = Self::create_ubo(dev_ctx, size)?;
        bundle.pool = Self::create_descriptor_pool(&bundle.dev_ctx.handle, size)?;
        bundle.sets = Self::create_descriptor_sets(
            &bundle.dev_ctx.handle,
            &bundle.pool,
            &bundle.layout,
            size,
        )?;

        Ok(bundle)
    }

    fn create_ubo_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let ubo_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&ubo_binding);

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .context("Failed to create Descriptor Set Layout!")?
        };

        Ok(layout)
    }

    fn create_ubo(
        dev_ctx: Arc<DeviceContext>,
        size: usize,
    ) -> Result<(Vec<BufferBundle>, Vec<*mut c_void>)> {
        let mut buffers = Vec::with_capacity(size);
        let mut mapped_ptrs = Vec::with_capacity(size);

        for _ in 0..size {
            let buffer = BufferBundle::new(
                dev_ctx.clone(),
                size_of::<UniformBufferObject>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mapped_ptr = unsafe {
                dev_ctx.handle.map_memory(
                    buffer.memory,
                    0,
                    buffer.size,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .context("Failed to map Uniform Buffer memory!")?;

            buffers.push(buffer);
            mapped_ptrs.push(mapped_ptr);
        }

        Ok((buffers, mapped_ptrs))
    }

    fn create_descriptor_pool(device: &ash::Device, size: usize) -> Result<vk::DescriptorPool> {
        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(size as u32)];
        let pool_info =
            vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_size).max_sets(size as u32);

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .context("Failed to create Descriptor Pool!")?
        };

        Ok(pool)
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        pool: &vk::DescriptorPool,
        layout: &vk::DescriptorSetLayout,
        size: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![*layout; size];
        let alloc_info =
            vk::DescriptorSetAllocateInfo::default().descriptor_pool(*pool).set_layouts(&layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate Descriptor Sets!")?
        };

        Ok(descriptor_sets)
    }

    fn update_descriptor_set(device: &ash::Device, buffer: &vk::Buffer, set: &vk::DescriptorSet) {
        let buffer_info =
            [vk::DescriptorBufferInfo::default().buffer(*buffer).offset(0).range(vk::WHOLE_SIZE)];
        let descriptor_write = [vk::WriteDescriptorSet::default()
            .dst_set(*set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info)];

        unsafe { device.update_descriptor_sets(&descriptor_write, &[]) };
    }
}

impl Drop for UniformBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_descriptor_pool(self.pool, None);
            self.dev_ctx.handle.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

struct FrameBundle {
    dev_ctx: Arc<DeviceContext>,

    s_image_available: vk::Semaphore,
    s_render_finished: vk::Semaphore,
    f_in_flight: vk::Fence,

    cmd_buffer: vk::CommandBuffer,
    descriptor_set: vk::DescriptorSet,
    u_buffer_mapped: *mut c_void,
}

impl FrameBundle {
    fn new(
        dev_ctx: Arc<DeviceContext>,
        cmd_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        u_buffer_mapped: *mut c_void,
    ) -> Result<Self> {
        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            s_image_available: vk::Semaphore::null(),
            s_render_finished: vk::Semaphore::null(),
            f_in_flight: vk::Fence::null(),
            cmd_buffer,
            descriptor_set,
            u_buffer_mapped,
        };

        unsafe {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            bundle.s_image_available = dev_ctx
                .handle
                .create_semaphore(&semaphore_info, None)
                .context("Failed to create Image Available Semaphore!")?;
            bundle.s_render_finished = dev_ctx
                .handle
                .create_semaphore(&semaphore_info, None)
                .context("Failed to create Render Finished Semaphore!")?;
            bundle.f_in_flight = dev_ctx
                .handle
                .create_fence(&fence_info, None)
                .context("Failed to create In Flight Fence!")?;
        }

        Ok(bundle)
    }

    pub fn update_uniform(&self, delta: f32, extent: vk::Extent2D) {
        let model = Matrix4::from_axis_angle(Vector3::unit_z(), Deg(360.0) * delta);
        let view = Matrix4::look_at_rh(
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let mut proj =
            cgmath::perspective(Deg(45.0), extent.width as f32 / extent.height as f32, 0.1, 10.0);
        proj[1][1] *= -1.0;
        let ubo = UniformBufferObject { model, view, proj };

        unsafe {
            std::ptr::copy_nonoverlapping(
                &ubo as *const UniformBufferObject,
                self.u_buffer_mapped as *mut UniformBufferObject,
                1,
            );
        }
    }
}

impl Drop for FrameBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_semaphore(self.s_image_available, None);
            self.dev_ctx.handle.destroy_semaphore(self.s_render_finished, None);
            self.dev_ctx.handle.destroy_fence(self.f_in_flight, None);
        }
    }
}

struct Engine {
    dev_ctx: Arc<DeviceContext>,

    _ubo: UniformBundle,
    pipeline: PipelineBundle,
    swapchain: SwapchainBundle,
    _command: CommandBundle,

    geometry: GeometryBundle,
    frames: Vec<FrameBundle>,
    current_frame: usize,
    pub framebuffer_resized: bool,
}

impl Engine {
    pub fn new(window: &Window, required_extensions: &[*const i8]) -> Result<Self> {
        let vk_ctx = Arc::new(VulkanContext::init(window, required_extensions)?);
        let dev_ctx = Arc::new(DeviceContext::init(vk_ctx.clone())?);

        let ubo = UniformBundle::new(dev_ctx.clone(), MAX_FRAMES_IN_FLIGHT)?;
        let pipeline = PipelineBundle::new(dev_ctx.clone(), &ubo.layout)?;
        let swapchain = SwapchainBundle::new(dev_ctx.clone())?;
        let command = CommandBundle::new(dev_ctx.clone(), MAX_FRAMES_IN_FLIGHT)?;

        let geometry = GeometryBundle::new(dev_ctx.clone(), &command, &VERTICES, &INDICES)?;
        let frames: Vec<_> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|idx| {
                UniformBundle::update_descriptor_set(
                    &dev_ctx.handle,
                    &ubo.buffers[idx].buffer,
                    &ubo.sets[idx],
                );
                FrameBundle::new(
                    dev_ctx.clone(),
                    command.buffers[idx],
                    ubo.sets[idx],
                    ubo.mapped_ptrs[idx],
                )
            })
            .collect::<Result<_>>()?;

        Ok(Engine {
            dev_ctx,
            _ubo: ubo,
            pipeline,
            swapchain,
            _command: command,
            geometry,
            frames,
            current_frame: 0,
            framebuffer_resized: false,
        })
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        let frame = &self.frames[self.current_frame];

        unsafe {
            let wait_fences = [frame.f_in_flight];
            self.dev_ctx
                .handle
                .wait_for_fences(&wait_fences, true, u64::MAX)
                .context("Failed to wait for In Flight Fence!")?;

            let (image_index, is_suboptimal) = match self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                frame.s_image_available,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                Err(err) => {
                    return Err(err).context("Failed to acquire next image from Swapchain!");
                }
            };

            let delta = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
                % 25_000) as f32
                / 25_000.0;

            frame.update_uniform(delta, self.swapchain.extent);

            self.dev_ctx
                .handle
                .reset_fences(&wait_fences)
                .context("Failed to reset In Flight Fence!")?;

            let cmd = [self.record_command(image_index, frame.cmd_buffer)?];
            let wait_semaphores = [frame.s_image_available];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [frame.s_render_finished];
            let submit_info = [vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd)
                .signal_semaphores(&signal_semaphores)];
            self.dev_ctx
                .handle
                .queue_submit(self.dev_ctx.graphics_queue, &submit_info, frame.f_in_flight)
                .context("Failed to submit draw command buffer!")?;

            let swapchains = [self.swapchain.handle];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            let changed = match self
                .swapchain
                .loader
                .queue_present(self.dev_ctx.present_queue, &present_info)
            {
                Ok(suboptimal) => suboptimal || is_suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => true,
                Err(err) => {
                    return Err(err).context("Failed to present Swapchain image!");
                }
            };

            if changed || self.framebuffer_resized {
                self.framebuffer_resized = false;
                self.recreate_swapchain()?;
            }

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        unsafe {
            self.dev_ctx.handle.device_wait_idle().context("Failed to wait device idle!")?;
        }
        self.swapchain = SwapchainBundle::new(self.dev_ctx.clone())?;

        Ok(())
    }

    fn record_command(
        &self,
        image_index: u32,
        cmd: vk::CommandBuffer,
    ) -> Result<vk::CommandBuffer> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.dev_ctx
                .handle
                .begin_command_buffer(cmd, &begin_info)
                .context("Failed to begin recording Command Buffer!")?;

            let image_memory_barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.swapchain.images[image_index as usize])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(&image_memory_barrier);
            self.dev_ctx.sync2.cmd_pipeline_barrier2(cmd, &dependency_info);

            let color_attachment_info = [vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain.image_views[image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                })];

            let render_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachment_info);
            self.dev_ctx.dyn_render.cmd_begin_rendering(cmd, &render_info);
            self.dev_ctx.handle.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            self.dev_ctx.handle.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.swapchain.extent.width as f32,
                    height: self.swapchain.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.dev_ctx.handle.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                }],
            );
            self.dev_ctx.handle.cmd_bind_vertex_buffers(
                cmd,
                0,
                &[self.geometry.v_buffer.buffer],
                &[0],
            );
            self.dev_ctx.handle.cmd_bind_index_buffer(
                cmd,
                self.geometry.i_buffer.buffer,
                0,
                vk::IndexType::UINT16,
            );
            let frame = &self.frames[self.current_frame];
            self.dev_ctx.handle.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[frame.descriptor_set],
                &[],
            );
            self.dev_ctx.handle.cmd_draw_indexed(cmd, INDICES.len() as u32, 1, 0, 0, 0);
            self.dev_ctx.dyn_render.cmd_end_rendering(cmd);

            let image_memory_barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::empty())
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(self.swapchain.images[image_index as usize])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(&image_memory_barrier);
            self.dev_ctx.sync2.cmd_pipeline_barrier2(cmd, &dependency_info);
            self.dev_ctx
                .handle
                .end_command_buffer(cmd)
                .context("Failed to end recording Command Buffer!")?;
        }
        Ok(cmd)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.device_wait_idle().ok();
        }
    }
}

#[derive(Default)]
struct HelloTriangleApplication {
    state: Option<(Engine, Window)>,
}

impl ApplicationHandler for HelloTriangleApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let result = (|| -> Result<(Engine, Window)> {
            let window = event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Vulkan tutorial with Ash")
                        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT)),
                )
                .context("Failed to create window!")?;
            let required_extensions = ash_window::enumerate_required_extensions(
                window.display_handle().unwrap().as_raw(),
            )
            .context("Failed to enumerate required extensions for surface creation!")?;
            Ok((
                Engine::new(&window, &required_extensions)
                    .context("Failed to initialize Vulkan Engine!")?,
                window,
            ))
        })();

        match result {
            Ok((engine, window)) => {
                self.state = Some((engine, window));
            }
            Err(e) => {
                eprintln!("Failed to initialize application: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some((engine, _)) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => {
                engine.framebuffer_resized = true;
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some((engine, _)) = self.state.as_mut() else { return };

        if let Err(e) = engine.draw_frame() {
            eprintln!("Failed to draw frame: {}", e);
        }
    }
}

fn main() {
    // If running under WSL, force winit to use X11 (Wayland support on WSL is unreliable).
    #[cfg(target_os = "linux")]
    {
        fn is_wsl() -> bool {
            if std::env::var_os("WSL_DISTRO_NAME").is_some() {
                return true;
            }
            if let Ok(ver) = std::fs::read_to_string("/proc/version") {
                let v = ver.to_lowercase();
                if v.contains("microsoft") || v.contains("wsl") {
                    return true;
                }
            }
            if let Ok(release) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
                let r = release.to_lowercase();
                if r.contains("microsoft") || r.contains("wsl") {
                    return true;
                }
            }
            false
        }

        if is_wsl() {
            unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "x11") };
            unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
            print!("[diagnostic] Detected WSL: forcing WINIT_UNIX_BACKEND=x11 ");
            println!("and unsetting WAYLAND_DISPLAY");
        }
    }

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = HelloTriangleApplication::default();
    let _ = event_loop.run_app(&mut app);
}
