use crate::config::{F, PI};

pub struct Field4D {
    pub amplitude: F,
    pub omega: F,
    pub N: F,
    pub x_envelop: F,
}

impl Field4D {
    pub fn new(amplitude: F, omega: F, N: F, x_envelop: F) -> Self {
        Self {
            amplitude,
            omega,
            N,
            x_envelop,
        }
    }

    pub fn electric_field_time_dependence(&self, t: F) -> [F; 4] {
        // Возвращает электрическое поле в момент времени t вдоль
        // каждой из пространственных осей x0, x1 и т.д.: массив размерности dim.
        // Каждый элемент этого массива содержит электрическое поле
        // в момент времени t вдоль соответствующей оси.
        // Например, E0 = electric_fielf(2.)[0] - электрическое
        // поле в момент времени t=2 вдоль оси x0.

        let mut ex: F = 0.0;
        let mut ey: F = 0.0;

        if 2. * PI * self.N / self.omega - t > 0. {
            ex = -self.amplitude
                * F::sin(self.omega * t / (2. * self.N)).powi(2)
                * F::sin(self.omega * t);
            ey = -self.amplitude
                * F::sin(self.omega * t / (2. * self.N)).powi(2)
                * F::cos(self.omega * t);
        }
        [ex, ey, ex, ey]
    }

    pub fn electric_field(&self, t: F, x: [F; 4]) -> [F; 4] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [
            self.electric_field_time_dependence(t)[0],
            self.electric_field_time_dependence(t)[1],
            self.electric_field_time_dependence(t)[2],
            self.electric_field_time_dependence(t)[3],
        ]
    }

    pub fn scalar_potential(&self, t: F, x: [F; 4]) -> F {
        -self.electric_field_time_dependence(t)[0] * x[0]
            - self.electric_field_time_dependence(t)[1] * x[1]
            - self.electric_field_time_dependence(t)[2] * x[2]
            - self.electric_field_time_dependence(t)[3] * x[3]
    }

    // Векторный потенциал
    pub fn vec_pot(&self, t: F) -> [F; 4] {
        let mut vec_pot_x: F = 0.0;
        let mut vec_pot_y: F = 0.0;
        let compute_vec_pot_x = |tau: F| {
            self.amplitude
                * (-1.0
                    + F::cos(tau) * (1.0 - self.N.powi(2) + self.N.powi(2) * F::cos(tau / self.N))
                    + self.N * F::sin(tau) * F::sin(tau / self.N))
                / (2.0 * self.omega * (-1.0 + self.N.powi(2)))
        };
        let compute_vec_pot_y = |tau: F| {
            self.amplitude
                * (F::sin(tau) * (-1.0 + self.N.powi(2) - self.N.powi(2) * F::cos(tau / self.N))
                    + self.N * F::cos(tau) * F::sin(tau / self.N))
                / (2.0 * self.omega * (-1.0 + self.N.powi(2)))
        };

        let tau: F = self.omega * t;
        let period: F = 2. * PI * self.N / self.omega;
        if t < period {
            vec_pot_x = compute_vec_pot_x(tau);
            vec_pot_y = compute_vec_pot_y(tau);
        } else {
            vec_pot_x = compute_vec_pot_x(period);
            vec_pot_y = compute_vec_pot_y(period);
        }
        [vec_pot_x, vec_pot_y, vec_pot_x, vec_pot_y]

        // if 2. * PI * self.N / self.omega - t > 0. {
        //     vec_pot[0] = self.amplitude / self.omega
        //         * F::sin(self.omega * t / (2. * self.N)).powi(2)
        //         * F::sin(self.omega * t);
        //     vec_pot[1] = self.amplitude / self.omega
        //         * F::sin(self.omega * t / (2. * self.N)).powi(2)
        //         * F::cos(self.omega * t);
        // }
        // vec_pot
    }

    // интеграл от векторного потенциала
    pub fn a(&self, t: F) -> [F; 4] {
        let tau: F = self.omega * t;
        // let mut a: [F; 2] = [0.0, 0.0];

        // a[0] = self.amplitude
        //     * t
        //     * (-1.0
        //         + F::cos(tau) * (1.0 - self.N.powi(2) + self.N.powi(2) * F::cos(tau / self.N))
        //         + self.N * F::sin(tau) * F::sin(tau / self.N))
        //     / (2.0 * (-1.0 + self.N.powi(2)) * self.omega);
        let ax: F = self.amplitude
            * (-(-1.0 + self.N.powi(2)) * tau
                + F::sin(tau)
                    * (-(-1.0 + self.N.powi(2)).powi(2)
                        + (self.N.powi(2) + self.N.powi(4)) * F::cos(tau / self.N))
                - 2.0 * self.N.powi(3) * F::cos(tau) * F::sin(tau / self.N))
            / (2.0 * (self.N.powi(2) - 1.0).powi(2) * self.omega.powi(2));

        let ay: F = self.amplitude
            * (1.0 - 3.0 * self.N.powi(2)
                + F::cos(tau)
                    * (-(-1.0 + self.N.powi(2)).powi(2)
                        + (self.N.powi(2) + self.N.powi(4)) * F::cos(tau / self.N))
                + 2.0 * self.N.powi(3) * F::sin(tau) * F::sin(tau / self.N))
            / (2.0 * (self.N.powi(2) - 1.0).powi(2) * self.omega.powi(2));

        [ax, ay, ax, ay]
    }

    // интеграл от квадрата векторного потенциала
    pub fn b(&self, t: F) -> F {
        let tau: F = self.omega * t;

        let term1: F =
            self.amplitude.powi(2) / (16.0 * (-1.0 + self.N.powi(2)).powi(3) * self.omega.powi(3));
        let term2: F = 8.0
            * (-(-1.0 + self.N.powi(2)).powi(2)
                + (self.N.powi(2) + self.N.powi(4)) * F::cos(tau / self.N))
            * F::sin(tau);
        let term3: F = 8.0
            * self.N.powi(3)
            * ((-1.0 + self.N.powi(2)).powi(2) - 2.0 * F::cos(tau))
            * F::sin(tau / self.N);
        let term4: F = (-1.0 + self.N.powi(2))
            * (2.0 * (4.0 - 3.0 * self.N.powi(2) + 3.0 * self.N.powi(4)) * tau
                + self.N.powi(3) * (-1.0 + self.N.powi(2)) * F::sin(2.0 * tau / self.N));
        term1 * (-term2 - term3 + term4)
    }
}
