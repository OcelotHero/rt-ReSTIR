extern crate winit;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use winit::dpi::LogicalSize;

use ash::khr::portability_enumeration::*;
use ash::vk::InstanceCreateFlags;

// use ash::version::IntanceV1_0;

#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        KHR_PORTABILITY_ENUMERATION_NAME.name().as_ptr(),
    ]
}

#[cfg(not(target_os = "macos"))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![]
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[derive(Default)]
struct HelloTriangleApplication {
    _entry: Option<ash::Entry>,
    window: Option<Window>,
    instance: Option<ash::Instance>,
}

impl ApplicationHandler for HelloTriangleApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan tutorial with Ash")
                    .with_inner_size(LogicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.window = Some(window);
        self.init_vulkan();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

impl HelloTriangleApplication {
    fn init_vulkan(&mut self) {
        let entry = ash::Entry::linked();
        self._entry = Some(entry);

        let app_name = std::ffi::CString::new("Hello Triangle").unwrap();
        let engine_name = std::ffi::CString::new("No Engine").unwrap();

        let app_info = ash::vk::ApplicationInfo {
            s_type: ash::vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: ash::vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: ash::vk::make_api_version(0, 1, 0, 0),
            api_version: ash::vk::make_api_version(0, 1, 0, 0),
            _marker: std::marker::PhantomData,
        };

        let extension_names = required_extension_names();

        let create_info = ash::vk::InstanceCreateInfo {
            s_type: ash::vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: (
                InstanceCreateFlags::default()
                | InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            ),
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let instance = unsafe {
            self._entry
                .as_ref()
                .unwrap()
                .create_instance(&create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        self.instance = Some(instance);
    }

    fn main_loop(&mut self, _event_loop: &ActiveEventLoop) {
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            self.instance.as_ref().unwrap().destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = HelloTriangleApplication::default();
    let _ = event_loop.run_app(&mut app);
}
