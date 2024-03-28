extern crate rustboy;

#[cfg(feature = "debug")]
mod debug;

use rustboy::*;

use cpal;
use clap::{clap_app, crate_version};
use wgpu::util::DeviceExt;
use winit::{
    dpi::{
        Size, LogicalSize
    },
    event::{
        Event, WindowEvent,
        ElementState,
        VirtualKeyCode
    },
    event_loop::EventLoop,
    window::WindowBuilder
};
use cpal::traits::StreamTrait;
use std::sync::Arc;


const FRAME_TIME: i64 = 16_666;
//const FRAME_TIME: i64 = 16_743; // 59.73 fps

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct Vertex {
    position:   [f32; 2],
    tex_coord:  [f32; 2]
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

fn main() {
    let app = clap_app!(rustboy =>
        (version: crate_version!())
        (author: "Simon Cooper")
        (about: "Game Boy and Game Boy Color emulator.")
        (@arg CART: "The path to the game cart to use.")
        (@arg debug: -d "Enter debug mode.")
        (@arg mute: -m "Mutes the emulator.")
        (@arg palette: -p +takes_value "Choose a palette. 'g' selects the classic green scheme, 'bw' forces greyscale. By default SGB colour will be used if available.")
        (@arg save: -s +takes_value "Save file path.")
    );

    let cmd_args = app.get_matches();

    let mute = cmd_args.is_present("mute");

    let cart = match cmd_args.value_of("CART") {
        Some(c) => c.to_string(),
        None => panic!("Usage: rustboy [cart name]. Run with --help for more options."),
    };

    let save_file = match cmd_args.value_of("save") {
        Some(c) => c.to_string(),
        None => make_save_name(&cart),
    };

    let palette = choose_palette(cmd_args.value_of("palette"));

    // Video
    let event_loop = EventLoop::new();//.expect("could not create event loop");

    let mut rustboy = RustBoy::new(&cart, &save_file, palette);

    //let mut averager = avg::Averager::<i64>::new(60);
    let mut frame_tex = [255_u8; 160 * 144 * 4];

    if cmd_args.is_present("debug") {
        #[cfg(feature = "debug")]
        debug::debug_mode(&mut rustboy);
    } else {
        let window = WindowBuilder::new()
            .with_inner_size(Size::Logical(LogicalSize{width: 320.0, height: 288.0}))
            .with_title("Super Rust Boy")
            .build(&event_loop)
            .expect("Couldn't create window");

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = unsafe {instance.create_surface(&window)}.expect("could not create surface");

        let adapter = futures::executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })).expect("Failed to find appropriate adapter");

        let (device, queue) = futures::executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default()
        }, None)).expect("Failed to create device");


        let size = window.inner_size();
        let mut surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb]
        };
        surface.configure(&device, &surface_config);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None
                },
            ]
        });
    
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[]
        });
    
        let texture_extent = wgpu::Extent3d {
            width: 160,
            height: 144,
            depth_or_array_layers: 1
        };
    
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: None,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb]
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::FilterMode::Nearest,
            ..Default::default()
        });
    
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler)
                }
            ],
            label: None
        });
    
        let vertices = vec![
            Vertex{position: [-1.0, -1.0], tex_coord: [0.0, 1.0]},
            Vertex{position: [1.0, -1.0], tex_coord: [1.0, 1.0]},
            Vertex{position: [-1.0, 1.0], tex_coord: [0.0, 0.0]},
            Vertex{position: [1.0, 1.0], tex_coord: [1.0, 0.0]},
        ];
    
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX
        });
    
        let module = device.create_shader_module(wgpu::include_wgsl!("./shaders/shader.wgsl"));
    
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 4 * 2,
                            shader_location: 1,
                        },
                    ]
                }]
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                .. Default::default()
                /*front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,*/
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            }),
            multiview: None
        });

        let audio_stream = make_audio_stream(&mut rustboy);
        if !mute {
            audio_stream.play().expect("Couldn't start audio stream");
        }

        let mut last_frame_time = chrono::Utc::now();
        let nanos = 1_000_000_000 / 60;
        let frame_time = chrono::Duration::nanoseconds(nanos);

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::LoopDestroyed => (),
                Event::MainEventsCleared => {
                    let now = chrono::Utc::now();
                    let since_last = now.signed_duration_since(last_frame_time);
                    if since_last < frame_time {
                        return;
                    }
                    last_frame_time = now;

                    rustboy.frame(&mut frame_tex);

                    queue.write_texture(
                        texture.as_image_copy(),
                        &frame_tex, 
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * texture_extent.width),
                            rows_per_image: None,
                        },
                        texture_extent
                    );

                    let frame = surface.get_current_texture().expect("Timeout when acquiring next swapchain tex.");
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: None});

                    {
                        let view = frame.texture.create_view(&Default::default());
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None
                        });
                        rpass.set_pipeline(&render_pipeline);
                        rpass.set_bind_group(0, &bind_group, &[]);
                        rpass.set_vertex_buffer(0, vertex_buf.slice(..));
                        rpass.draw(0..4, 0..1);
                    }

                    queue.submit([encoder.finish()]);
                    frame.present();
                },
                Event::WindowEvent {
                    window_id: _,
                    event: w,
                } => match w {
                    WindowEvent::CloseRequested => {
                        ::std::process::exit(0);
                    },
                    WindowEvent::KeyboardInput {
                        device_id: _,
                        input: k,
                        is_synthetic: _,
                    } => {
                        let pressed = match k.state {
                            ElementState::Pressed => true,
                            ElementState::Released => false,
                        };
                        match k.virtual_keycode {
                            Some(VirtualKeyCode::X)         => rustboy.set_button(Button::A, pressed),
                            Some(VirtualKeyCode::Z)         => rustboy.set_button(Button::B, pressed),
                            Some(VirtualKeyCode::Space)     => rustboy.set_button(Button::Select, pressed),
                            Some(VirtualKeyCode::Return)    => rustboy.set_button(Button::Start, pressed),
                            Some(VirtualKeyCode::Up)        => rustboy.set_button(Button::Up, pressed),
                            Some(VirtualKeyCode::Down)      => rustboy.set_button(Button::Down, pressed),
                            Some(VirtualKeyCode::Left)      => rustboy.set_button(Button::Left, pressed),
                            Some(VirtualKeyCode::Right)     => rustboy.set_button(Button::Right, pressed),
                            _ => {},
                        }
                    },
                    WindowEvent::Resized(size) => {
                        surface_config.width = size.width;
                        surface_config.height = size.height;
                        surface.configure(&device, &surface_config);
                    },
                    _ => {}
                },
                //Event::RedrawRequested(_) => {},
                _ => {},
            }
        });
    }
}

fn make_save_name(cart_name: &str) -> String {
    match cart_name.find(".") {
        Some(pos) => cart_name[0..pos].to_string() + ".sav",
        None      => cart_name.to_string() + ".sav"
    }
}

fn choose_palette(palette: Option<&str>) -> UserPalette {
    match palette {
        Some(s) => match s {
            "g" => UserPalette::Classic,
            "bw" => UserPalette::Greyscale,
            _ => UserPalette::Default
        },
        None => UserPalette::Default
    }
}

fn pick_output_config(device: &cpal::Device) -> cpal::SupportedStreamConfigRange {
    use cpal::traits::DeviceTrait;

    const MIN: u32 = 32_000;

    let supported_configs_range = device.supported_output_configs()
        .expect("error while querying configs");

    for config in supported_configs_range {
        let cpal::SampleRate(v) = config.max_sample_rate();
        if v >= MIN {
            return config;
        }
    }

    device.supported_output_configs()
        .expect("error while querying formats")
        .next()
        .expect("No supported config")
}

fn make_audio_stream(rustboy: &mut RustBoy) -> cpal::Stream {
    use cpal::traits::{
        DeviceTrait,
        HostTrait
    };

    let host = cpal::default_host();
    let device = host.default_output_device().expect("no output device available.");

    let config = pick_output_config(&device).with_max_sample_rate();
    let sample_rate = config.sample_rate().0 as usize;
    println!("Audio sample rate {}", sample_rate);
    let mut audio_handler = rustboy.enable_audio(sample_rate);

    device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _| {
            audio_handler.get_audio_packet(data);
        },
        move |err| {
            println!("Error occurred: {}", err);
        }
    ).unwrap()
}

/*
// Averager

use std::{
    collections::VecDeque,
    ops::{
        Add,
        Div
    }
};

pub struct Averager<T: Add + Div> {
    queue:      VecDeque<T>,
    max_len:    usize
}

impl Averager<i64> {
    pub fn new(len: usize) -> Self {
        Averager {
            queue:      VecDeque::with_capacity(len),
            max_len:    len
        }
    }

    pub fn add(&mut self, data: i64) {
        self.queue.push_back(data);

        if self.queue.len() > self.max_len {
            self.queue.pop_front();
        }
    }

    pub fn get_avg(&self) -> i64 {
        self.queue.iter().sum::<i64>() / (self.queue.len() as i64)
    }
}
*/