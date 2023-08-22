use egui::{Frame, Rect, Rgba, Rounding, Sense, Ui, Vec2};
use rand::Rng;

use crate::{
    kernels::{basic_mnca, larger_than_life_layered_kernel, life_layered_kernel, Life, KernelCache},
    sim::Dense, array2d::Array2D,
};

pub struct TemplateApp {
    sim: Dense,
    pause: bool,
    single_step: bool,
}

impl Default for TemplateApp {
    fn default() -> Self {
        //let kernel = Box::new(basic_mnca());
        let kernel = Box::new(KernelCache::new(Box::new(basic_mnca())));
        //let kernel = Box::new(KernelCache::new(Box::new(larger_than_life_layered_kernel())));
        let mut sim = Dense::new(kernel, 17*3, 10*3);

        let mut rng = rand::thread_rng();
        for block in sim.data_mut().data_mut() {
            for pixel in block.data_mut() {
                *pixel = rng.gen_bool(0.5);
            }
        }

        Self {
            sim,
            pause: true,
            single_step: false,
        }
    }
}

impl TemplateApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        if !self.pause || self.single_step {
            self.sim.step();
            self.single_step = false;
        }

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.pause, "Pause");
                self.single_step |= ui.button("Step").clicked();
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                sim_widget(&mut self.sim, ui);
            });
        });
    }
}

/// Maps sim coordinates to/from egui coordinates
struct CoordinateMapping {
    width: f32,
    height: f32,
    area: Rect,
}

impl CoordinateMapping {
    pub fn new(width: usize, height: usize, area: Rect) -> Self {
        Self {
            width: width as f32,
            height: height as f32,
            area,
        }
    }

    pub fn sim_to_egui(&self, pt: (usize, usize)) -> egui::Pos2 {
        let (x, y) = pt;
        egui::Pos2::new(
            (x as f32 / self.width) * self.area.width(),
            (1. - y as f32 / self.height) * self.area.height(),
        ) + self.area.min.to_vec2()
    }

    pub fn sim_to_egui_vect(&self, pt: (usize, usize)) -> egui::Vec2 {
        let (x, y) = pt;
        Vec2::new(
            (x as f32 / self.width) * self.area.width(),
            (y as f32 / self.height) * self.area.height(),
        )
    }

    /*
    pub fn egui_to_sim(&self, pt: egui::Pos2) -> (usize, usize) {
        glam::Vec2::new(
            (pt.x / self.area.width()) * self.width,
            (1. - pt.y / self.area.height()) * self.height,
        )
    }
    */
}

fn sim_widget(sim: &mut Dense, ui: &mut Ui) {
    let (widget_area, _response) =
        ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

    let (w, h) = sim.pixel_dims();
    let coords = CoordinateMapping::new(w, h, widget_area);

    let rect_size = coords.sim_to_egui_vect((1, 1));

    // Draw particles
    let painter = ui.painter_at(widget_area);
    for j in 0..h {
        for i in 0..w {
            if sim.get_pixel((i, j)) {
                let pt = coords.sim_to_egui((i, j));
                //dbg!(pt);
                let rect = Rect::from_min_size(pt, rect_size);
                painter.rect_filled(rect, Rounding::none(), Rgba::WHITE);
            }
        }
    }
}