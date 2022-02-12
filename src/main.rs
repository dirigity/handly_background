use rand::{prelude::ThreadRng, Rng};
use std::ops::{Add, Sub};

#[cfg(windows)]
extern crate winapi;

#[cfg(windows)]
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("C:\\Users\\Jaime\\Desktop\\Proyects\\_handly_background\\cpp\\tri_ploter.cpp");

        fn draw_tri(
            x1: i32,
            y1: i32,
            x2: i32,
            y2: i32,
            x3: i32,
            y3: i32,
            maxX: i32,
            maxY: i32,
            r: i32,
            g: i32,
            b: i32,
        );
    }
}

use angular::atan2;
use std::{cmp::Ordering, time::Instant};
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Debug, Clone, Copy)]
struct Ver {
    anchor: Point<f32>,
    radius: f32,
    rotation_speed: f32,
    amplitude_speed: f32,
}

trait Geometry {
    fn distance(&self, other: Self) -> f32;
    fn mul(&self, m: f32) -> Point<f32>;
    fn perp(&self) -> Point<f32>;
    fn angle(&self) -> f32;
    fn new(x: f32, y: f32) -> Self;
}

impl Add for Point<f32> {
    type Output = Point<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Point<f32> {
    type Output = Point<f32>;

    fn sub(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Geometry for Point<f32> {
    fn new(x: f32, y: f32) -> Self {
        Self { x: x, y: y }
    }
    fn distance(&self, other: Self) -> f32 {
        (((self.x - other.x).powf(2.) + (self.y - other.y).powf(2.)) as f32).sqrt()
    }
    fn mul(&self, m: f32) -> Point<f32> {
        Point::new(self.x * m, self.y * m)
    }
    fn perp(&self) -> Point<f32> {
        Point::new(self.y, -self.x)
    }
    fn angle(&self) -> f32 {
        atan2(self.y as f32, self.x as f32).in_radians()
    }
}
#[derive(Debug, Clone)]
struct Tri {
    v: Vec<Point<f32>>,
}

fn line_intersection(
    p_a_in: &Point<f32>,
    v_a_in: &Point<f32>,
    p_b_in: &Point<f32>,
    v_b_in: &Point<f32>,
) -> Point<f32> {
    let mut v_a = *v_a_in;
    let v_b = *v_b_in;
    let p_a = *p_a_in;
    let p_b = *p_b_in;

    if v_a.y == 0. {
        v_a.y = 0.001;
        v_a.x *= 100.;
    }

    let mut tmp = v_b.x - (v_b.y * v_a.x) / (v_a.y);

    if tmp == 0. {
        tmp = 0.00001;
    }

    let b = (p_a.x + (p_b.y - p_a.y) * v_a.x / v_a.y - p_b.x) / tmp;

    p_b + v_b.mul(b as f32)
}

impl Tri {
    fn should_contain(&self, v: Point<f32>) -> bool {
        self.circ_center().distance(v) <= self.circ_radius()
    }

    fn circ_center(&self) -> Point<f32> {
        let point_a = (self.v[0] + self.v[1]).mul(0.5);
        let point_b = (self.v[2] + self.v[1]).mul(0.5);
        let vec_a = (self.v[0] - self.v[1]).perp();
        let vec_b = (self.v[2] - self.v[1]).perp();

        line_intersection(&point_a, &vec_a, &point_b, &vec_b)
    }
    fn circ_radius(&self) -> f32 {
        self.circ_center().distance(self.v[0])
    }

    fn divide(&self, v: Point<f32>) -> Vec<Tri> {
        let mut ret = Vec::new();
        let mut color_point_angle: Vec<(Point<f32>, f32)> =
            self.v.iter().map(|e| (*e, (*e - v).angle())).collect();

        // println!("{:?},{:?}", color_point_angle, self.v);

        color_point_angle.sort_by(|(_, a), (_, b)| {
            if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        // println!("{:?}", color_point_angle);

        for i in 0..color_point_angle.len() {
            let (p1_v, _) = color_point_angle[i];
            let (p2_v, _) = color_point_angle[(i + 1) % color_point_angle.len()];

            ret.push(Tri {
                v: vec![v, p1_v, p2_v],
            })
        }

        ret
    }

    // fn cut(&self, p: Point<f32>, v: Point<f32>) -> Option<Vec<Tri>> {
    //     let side_a = (v.angle() - (p - self.v[0]).angle()).signum();
    //     let side_b = (v.angle() - (p - self.v[1]).angle()).signum();
    //     let side_c = (v.angle() - (p - self.v[2]).angle()).signum();

    //     if side_a != side_b || side_b != side_c {
    //         let mut tmp = vec![
    //             (side_a, self.v[0]),
    //             (side_b, self.v[1]),
    //             (side_c, self.v[2]),
    //         ];

    //         tmp.sort_by(|(a, _), (b, _)| {
    //             if a > b {
    //                 Ordering::Greater
    //             } else {
    //                 Ordering::Less
    //             }
    //         });

    //         if tmp[0].0 == tmp[1].0 {
    //             tmp.swap(0, 2);
    //         }

    //         // tmp = [alone, togu, ether]

    //         let cut_0_1 = line_intersection(&p, &v, &tmp[0].1, &(tmp[0].1 - tmp[1].1));
    //         let cut_0_2 = line_intersection(&p, &v, &tmp[0].1, &(tmp[0].1 - tmp[2].1));

    //         let ret = vec![
    //             Tri {
    //                 v: vec![tmp[0].1, cut_0_1, cut_0_2],
    //             },
    //             Tri {
    //                 v: vec![tmp[1].1, cut_0_1, cut_0_2],
    //             },
    //             Tri {
    //                 v: vec![tmp[1].1, tmp[2].1, cut_0_2],
    //             },
    //         ];

    //         Some(ret)
    //     } else {
    //         None
    //     }
    // }

    fn merge(&self, other: Tri) -> Tri {
        let mut ret = self.clone();
        ret.v.extend(&other.v);

        ret.v.sort_by(|a, b| {
            if a.x == b.x {
                if a.y == b.y {
                    Ordering::Equal
                } else if a.y > b.y {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            } else {
                if a.x < b.x {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
        });
        ret.v.dedup();

        ret
    }
}
fn teselate(sw: f32, sh: f32, positions: impl Iterator<Item = Point<f32>>) -> Vec<Tri> {
    // println!("starting to calc tris");

    let mut tris = Vec::new();
    tris.push(Tri {
        v: vec![
            Point::new(-10. * sw, -10. * sh),
            Point::new(20. * sw, 0.),
            Point::new(0., 20. * sh),
        ],
    });

    for point in positions {
        // println!("start loop");

        let mut new_tris = Vec::new();
        let mut big_tri = Tri { v: vec![] };

        // println!("a, {}", tris.len());

        for t in tris {
            if t.should_contain(point) {
                big_tri = big_tri.merge(t);
            } else {
                new_tris.push(t);
            }
        }

        let division = big_tri.divide(point);
        new_tris.extend(division);
        tris = new_tris;
    }

    tris = tris
        .iter()
        .filter(|e| {
            Point::new(e.v[0].x, e.v[0].y) != Point::new(e.v[2].x, e.v[2].y)
                && Point::new(e.v[0].x, e.v[0].y) != Point::new(e.v[1].x, e.v[1].y)
                && Point::new(e.v[1].x, e.v[1].y) != Point::new(e.v[2].x, e.v[2].y)
        })
        .map(|e| e.clone())
        .collect();

    // println!("c, {}", tris.len());

    tris
}

const GRID_W: i32 = 16;
const GRID_H: i32 = 9;

const WORLD_W: i32 = 1600;
const WORLD_H: i32 = 900;

const MEAN_RADIUS: f32 = ((WORLD_W / GRID_W + WORLD_H / GRID_H) / 4) as f32;

const POINTS_LEN: usize = ((GRID_W + 4) * (GRID_H + 4)) as usize;

const goal_fps_charged: f32 = 20.;
const goal_fps_charging: f32 = 13.;
const goal_fps_unplugued: f32 = 8.;
const low_batery_warning: f64 = 0.3;

fn main() {
    let manager = battery::Manager::new().unwrap();
    let mut wait: f32 = 0.;
    let mut rng = rand::thread_rng();
    let two_d_vertices: Vec<Vec<Ver>> = (-2..(GRID_W + 2))
        .map(|x| {
            let x_word = x as f32 / GRID_W as f32 * WORLD_W as f32;
            (-2..(GRID_H + 2))
                .map(|y| {
                    let y_word = y as f32 / GRID_H as f32 * WORLD_H as f32;

                    Ver {
                        anchor: Point::new(x_word, y_word),
                        amplitude_speed: rand_range(0.1, 0.4, &mut rng),
                        radius: if (x == 0 && y == 0)
                            || (x == 0 && y == GRID_W)
                            || (x == GRID_H && y == 0)
                            || (x == GRID_H && y == GRID_W)
                        {
                            0.
                        } else {
                            MEAN_RADIUS * rand_range(1.5, 3., &mut rng)
                        },
                        rotation_speed: rand_range(0.1, 0.4, &mut rng),
                    }
                })
                .collect()
        })
        .collect();

    let start = Instant::now();

    loop {
        let iteration_restart = Instant::now();
        let bulk_frames = 100;

        let maybe_battery = manager.batteries().unwrap().next().unwrap();
        let battery = maybe_battery.unwrap();
        let state = battery.state();

        let goal_fps = match state {
            battery::State::Full => goal_fps_charged,
            battery::State::Charging => goal_fps_charging,
            _ => goal_fps_unplugued,
        };

        for _ in 0..bulk_frames {
            let ms = 1000. + start.elapsed().as_millis() as f32 / 3000.;

            //calculate vertices
            let points = two_d_vertices.iter().flatten().map(|v| {
                v.anchor
                    + polar_to_euclidean(
                        v.radius * (ms * v.amplitude_speed).sin(),
                        ms * v.rotation_speed,
                    )
            });

            // teselate
            let mut teselation = teselate(WORLD_W as f32, WORLD_H as f32, points);
            teselation = teselation
                .iter()
                .filter(|e| in_bounds(e.v[0]) || in_bounds(e.v[1]) || in_bounds(e.v[2]))
                .map(|e| Tri {
                    v: vec![
                        to_bounds(e.v[0], WORLD_W, WORLD_H, &mut rng),
                        to_bounds(e.v[1], WORLD_W, WORLD_H, &mut rng),
                        to_bounds(e.v[2], WORLD_W, WORLD_H, &mut rng),
                    ],
                })
                .filter(|e| e.v[0] != e.v[1] && e.v[1] != e.v[2] && e.v[2] != e.v[0])
                .map(|e| e.clone())
                .collect();

            //ffi::draw_tri(0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0);
            //ffi::draw_tri(1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0);

            // raster
            let time_factor = (ms / 10.) % 1.;
            let a = hsv(time_factor * 360., 0.6, 1.);
            let b = hsv(time_factor * 360., 0.9, 0.7);
            let c = (b.0 * 0.3, b.1 * 0.3, b.2 * 0.3);

            for tri in teselation {
                let mean_point = tri.v.iter().fold(Point::new(0., 0.), |acc, elm| acc + *elm);

                // println!("({}) > {:?} : {:?}", time_factor * 360., a, b);

                let steps = vec![0., 0.5, 1.];
                let colors = vec![a, b, c];

                let (r, g, b) = color_gradient(mean_point.x / 3. / WORLD_W as f32, steps, colors);

                ffi::draw_tri(
                    tri.v[0].x as i32,
                    tri.v[0].y as i32,
                    tri.v[1].x as i32,
                    tri.v[1].y as i32,
                    tri.v[2].x as i32,
                    tri.v[2].y as i32,
                    WORLD_W,
                    WORLD_H,
                    r as i32,
                    g as i32,
                    b as i32,
                );
            }

            // fps controll
            std::thread::sleep(std::time::Duration::from_millis(wait as u64));
        }

        let time = iteration_restart.elapsed().as_millis() as f32;
        let time_per_frame = time / bulk_frames as f32;
        let fps = 1000. / time_per_frame;

        let goal_time_per_frame = 1000. / goal_fps;

        println!("fps: {}", fps);
        println!("    time_per_frame: {}", time_per_frame - wait);
        println!("    goal_time_per_frame: {}", goal_time_per_frame);

        wait = goal_time_per_frame - (time_per_frame - wait);

        println!("    wait: {}", wait);

        if wait < 0. {
            println!(" ----------------------------- ");
            println!("| unrealistic fps spectations |");
            println!(" ----------------------------- ");
            wait = 0.;
        }
    }
}

fn polar_to_euclidean(m: f32, a: f32) -> Point<f32> {
    Point::new(a.cos() * m, a.sin() * m)
}

fn rand_range(start: f32, end: f32, rng: &mut ThreadRng) -> f32 {
    start + (end - start) * rng.gen::<f32>() as f32
}

fn color_gradient(i_in: f32, steps: Vec<f32>, colors: Vec<(f32, f32, f32)>) -> (f32, f32, f32) {
    let i = i_in.min(1.).max(0.);

    let j = steps
        .iter()
        .position(|elm| elm >= &i)
        .unwrap_or(steps.len() - 1)
        .max(1);
    let first_weight = (i - steps[j - 1]) / (steps[j] - steps[j - 1]);
    let second_weight = 1. - first_weight;
    let first_index = j - 1;
    let second_index = j;

    (
        (colors[first_index].0 * second_weight) + (colors[second_index].0 * first_weight),
        (colors[first_index].1 * second_weight) + (colors[second_index].1 * first_weight),
        (colors[first_index].2 * second_weight) + (colors[second_index].2 * first_weight),
    )
}

fn in_bounds(p: Point<f32>) -> bool {
    let w = WORLD_W;
    let h = WORLD_H;
    let margin = 10.;
    p.x >= -margin && p.y >= -margin && p.x <= w as f32 + margin && p.y <= h as f32 + margin
}

fn to_bounds(p: Point<f32>, w: i32, h: i32, rng: &mut ThreadRng) -> Point<f32> {
    let mut rng_c = rng;
    Point::new(
        p.x.min(w as f32 - rand_range(0.01, 0.4, &mut rng_c))
            .max(rand_range(0.01, 0.4, &mut rng_c)),
        p.y.min(h as f32 - rand_range(0.01, 0.4, &mut rng_c))
            .max(rand_range(0.01, 0.4, &mut rng_c)),
    )
}

fn hsv(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let h_ = h / 60.;
    let x = c * (1. - ((h_ % 2.) - 1.).abs());

    // println!("x:{}", x);

    let (r_, g_, b_) = match h_.floor() as i32 {
        0 => (c, x, 0.),
        1 => (x, c, 0.),
        2 => (0., c, x),
        3 => (0., x, c),
        4 => (x, 0., c),
        5 => (c, 0., x),
        _ => (0., 0., 0.),
    };

    // println!("rgb:{:?}, h:{}", (r_, g_, b_), h_);

    let m = v - c;

    ((r_ + m) * 255., (g_ + m) * 255., (b_ + m) * 255.)
}
