use rand::{prelude::ThreadRng, Rng};
use std::{
    ops::{Add, Sub},
    thread::sleep,
};

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
    anchor: Point<f64>,
    radius: f64,
    rotation_speed: f64,
    amplitude_speed: f64,
}

trait Geometry {
    fn distance(&self, other: Self) -> f64;
    fn mul(&self, m: f64) -> Point<f64>;
    fn perp(&self) -> Point<f64>;
    fn angle(&self) -> f64;
    fn new(x: f64, y: f64) -> Self;
}

impl Add for Point<f64> {
    type Output = Point<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Point<f64> {
    type Output = Point<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Geometry for Point<f64> {
    fn new(x: f64, y: f64) -> Self {
        Self { x: x, y: y }
    }
    fn distance(&self, other: Self) -> f64 {
        (((self.x - other.x).powf(2.) + (self.y - other.y).powf(2.)) as f64).sqrt()
    }
    fn mul(&self, m: f64) -> Point<f64> {
        Point::new(self.x * m, self.y * m)
    }
    fn perp(&self) -> Point<f64> {
        Point::new(self.y, -self.x)
    }
    fn angle(&self) -> f64 {
        atan2(self.y as f64, self.x as f64).in_radians()
    }
}

#[derive(Debug, Clone)]
struct Tri {
    v: Vec<Point<f64>>,
}

fn line_intersection(
    p_a_in: &Point<f64>,
    v_a_in: &Point<f64>,
    p_b_in: &Point<f64>,
    v_b_in: &Point<f64>,
) -> Point<f64> {
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

    p_b + v_b.mul(b as f64)
}

impl Tri {
    fn should_contain(&self, v: Point<f64>) -> bool {
        self.circ_center().distance(v) <= self.circ_radius()
    }

    fn circ_center(&self) -> Point<f64> {
        let point_a = (self.v[0] + self.v[1]).mul(0.5);
        let point_b = (self.v[2] + self.v[1]).mul(0.5);
        let vec_a = (self.v[0] - self.v[1]).perp();
        let vec_b = (self.v[2] - self.v[1]).perp();

        line_intersection(&point_a, &vec_a, &point_b, &vec_b)
    }
    fn circ_radius(&self) -> f64 {
        self.circ_center().distance(self.v[0])
    }

    fn divide(&self, v: Point<f64>) -> Vec<Tri> {
        let mut ret = Vec::new();
        let mut color_point_angle: Vec<(Point<f64>, f64)> =
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

    // fn cut(&self, p: Point<f64>, v: Point<f64>) -> Option<Vec<Tri>> {
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
fn teselate(sw: f64, sh: f64, positions: &Vec<Point<f64>>) -> Vec<Tri> {
    // println!("starting to calc tris");

    let mut tris = Vec::new();
    tris.push(Tri {
        v: vec![
            Point::new(-10. * sw, -10. * sh),
            Point::new(20. * sw, 0.),
            Point::new(0., 20. * sh),
        ],
    });

    for i in 0..positions.len() {
        // println!("start loop");

        let mut new_tris = Vec::new();
        let mut big_tri = Tri { v: vec![] };

        // println!("a, {}", tris.len());

        for t in tris {
            if t.should_contain(positions[i]) {
                big_tri = big_tri.merge(t);
            } else {
                new_tris.push(t);
            }
        }

        let division = big_tri.divide(positions[i]);
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
fn main() {
    let mut rng = rand::thread_rng();
    let mut vertices = Vec::new();

    let grid_w = 16;
    let grid_h = 9;

    let word_w = 1600;
    let word_h = 900;

    let mean_radius = ((word_w / grid_w + word_h / grid_h) / 4) as f64;

    for x in -1..(grid_w + 2) {
        let x_word = x as f64 / grid_w as f64 * word_w as f64;
        for y in -1..(grid_h + 2) {
            let y_word = y as f64 / grid_h as f64 * word_h as f64;

            vertices.push(Ver {
                anchor: Point::new(x_word, y_word),
                amplitude_speed: rand_range(0.1, 0.4, &mut rng),
                radius: mean_radius * rand_range(1.5, 3., &mut rng),
                rotation_speed: rand_range(0.1, 0.4, &mut rng),
            });
        }
    }
    let start = Instant::now();

    loop {
        let iteration_restart = Instant::now();
        let ms = 1000. + start.elapsed().as_millis() as f64 / 3000.;

        let mut points = Vec::new();
        for v in &vertices {
            points.push(to_bounds(
                v.anchor
                    + polar_to_euclidean(
                        v.radius * (ms * v.amplitude_speed).sin(),
                        ms * v.rotation_speed,
                    ),
                word_w,
                word_h,
                &mut rng,
            ));
        }
        points.push(Point::new(0., 0.));
        points.push(Point::new(0., word_w as f64));
        points.push(Point::new(word_h as f64, 0.));
        points.push(Point::new(word_h as f64, word_w as f64));
        let mut teselation = teselate(word_w as f64, word_h as f64, &points);
        teselation = teselation
            .iter()
            .filter(|e| {
                in_bounds(e.v[0], word_w, word_h)
                    && in_bounds(e.v[1], word_w, word_h)
                    && in_bounds(e.v[2], word_w, word_h)
            })
            .map(|e| e.clone())
            .collect();

        // ffi::draw_tri(0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0);
        // ffi::draw_tri(1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0);

        for tri in teselation {
            let mean_point = tri.v.iter().fold(Point::new(0., 0.), |acc, elm| acc + *elm);
            let (r, g, b) = color_gradient(
                mean_point.x / 3. / word_w as f64, //(1. + perlin.get([mean_point.x / 1000., mean_point.y / 1000., ms / 50.])) / 2.,
            );

            ffi::draw_tri(
                tri.v[0].x as i32,
                tri.v[0].y as i32,
                tri.v[1].x as i32,
                tri.v[1].y as i32,
                tri.v[2].x as i32,
                tri.v[2].y as i32,
                word_w,
                word_h,
                r,
                g,
                b,
            );
        }

        sleep(iteration_restart.elapsed() * 3);
    }
}

fn polar_to_euclidean(m: f64, a: f64) -> Point<f64> {
    Point::new(a.cos() * m, a.sin() * m)
}

fn rand_range(start: f64, end: f64, rng: &mut ThreadRng) -> f64 {
    start + (end - start) * rng.gen::<f64>() as f64
}

fn color_gradient(i_in: f64) -> (i32, i32, i32) {
    let i = i_in.min(1.).max(0.);
    let a = [255., 200., 100.];
    //let a = Rgb([158u8, 228, 147]);
    //let b = Rgb([74u8, 123, 157]);
    //let c = Rgb([173u8, 122, 153]);
    let d = [217., 131., 36.];
    let e = [37., 41., 28.];
    let steps = vec![0., 1.]; //vec![0., 0.5, 1.];
    let colors = vec![a, e];

    let j = steps
        .iter()
        .position(|elm| elm >= &i)
        .unwrap_or(steps.len() - 1)
        .max(1);
    let first_weight = (i - steps[j - 1]) / (steps[j] - steps[j - 1]);
    let second_weight = 1. - first_weight;
    let first_index = j - 1;
    let second_index = j;

    // println!("{},{}", first_weight, second_weight);

    (
        ((colors[first_index][0] * second_weight) + (colors[second_index][0] * first_weight))
            as i32,
        ((colors[first_index][1] * second_weight) + (colors[second_index][1] * first_weight))
            as i32,
        ((colors[first_index][2] * second_weight) + (colors[second_index][2] * first_weight))
            as i32,
    )
}

fn in_bounds(p: Point<f64>, w: i32, h: i32) -> bool {
    p.x >= 0. && p.y >= 0. && p.x <= (w + 10) as f64 && p.y <= (h + 10) as f64
}

fn to_bounds(p: Point<f64>, w: i32, h: i32, rng: &mut ThreadRng) -> Point<f64> {
    let mut rng_c = rng;
    Point::new(
        p.x.min(w as f64 - rand_range(0.01, 0.02, &mut rng_c))
            .max(rand_range(0.01, 0.02, &mut rng_c)),
        p.y.min(h as f64 - rand_range(0.01, 0.02, &mut rng_c))
            .max(rand_range(0.01, 0.02, &mut rng_c)),
    )
}
