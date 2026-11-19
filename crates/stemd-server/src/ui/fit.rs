//! The window's height, following what its lists want.

/// Frames to wait for a resize before taking the height the window has.
const PATIENCE: u32 = 30;

/// A resize asked for and not yet seen.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Asked {
    target: f32,
    /// The window's height when it was asked.
    from: f32,
    frames: u32,
}

#[derive(Default)]
pub struct Fit {
    /// The wanted height the window was last sized for.
    wanted: Option<f32>,
    asked: Option<Asked>,
    /// How much taller the window came out than asked, when it was clamped.
    excess: f32,
}

impl Fit {
    /// Whether the window has been sized once.
    pub const fn started(&self) -> bool {
        self.wanted.is_some()
    }

    /// Size the window for the first time; returns the height to ask for, if any.
    pub fn start(&mut self, height: f32, wanted: f32, delta: f32) -> Option<f32> {
        self.wanted = Some(wanted);
        self.ask(height, delta)
    }

    /// Follow the lists' `wanted` height; returns the height to ask for and the layout growth.
    pub fn step(&mut self, height: f32, wanted: f32) -> (Option<f32>, f32) {
        if let Some(asked) = &mut self.asked {
            if (height - asked.from).abs() >= 0.5 || asked.frames >= PATIENCE {
                self.excess = height - asked.target;
                self.asked = None;
            } else {
                asked.frames += 1;
            }
        }
        let mut target = None;
        if let Some(previous) = self.wanted {
            target = self.ask(height, wanted - previous);
            if target.is_some() {
                self.wanted = Some(wanted);
            }
        }
        let grow = self.asked.map_or(0.0, |a| a.target - height);
        (target, grow)
    }

    /// Whether a resize is under way.
    pub const fn waiting(&self) -> bool {
        self.asked.is_some()
    }

    fn ask(&mut self, height: f32, delta: f32) -> Option<f32> {
        // Heights wobble by a fraction of a point as text is laid out, and a
        // window that resizes by half a pixel every frame never settles.
        if delta.abs() < 1.0 {
            return None;
        }
        let (base, from) = match self.asked {
            Some(asked) => (asked.target, asked.from),
            None => (height - self.excess, height),
        };
        let target = base + delta;
        self.excess = 0.0;
        self.asked = Some(Asked {
            target,
            from,
            frames: 0,
        });
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn started(height: f32, wanted: f32) -> Fit {
        let mut fit = Fit::default();
        assert_eq!(fit.start(height, wanted, 0.0), None);
        fit
    }

    #[test]
    fn the_layout_keeps_the_growth_until_the_window_gets_it() {
        let mut fit = started(600.0, 100.0);
        assert_eq!(fit.step(600.0, 190.0), (Some(690.0), 90.0));
        assert_eq!(fit.step(600.0, 190.0), (None, 90.0), "not applied yet");
        assert_eq!(fit.step(690.0, 190.0), (None, 0.0));
        assert!(!fit.waiting());
    }

    #[test]
    fn a_clamped_shrink_is_not_grown_back_twice() {
        let mut fit = started(690.0, 190.0);
        assert_eq!(fit.step(690.0, 100.0).0, Some(600.0));
        // Clamped at a 650-point minimum.
        assert_eq!(fit.step(650.0, 100.0), (None, 0.0));
        assert_eq!(
            fit.step(650.0, 190.0).0,
            Some(690.0),
            "grew from the clamped height"
        );
    }

    #[test]
    fn a_resize_the_system_ignores_is_given_up() {
        let mut fit = started(600.0, 100.0);
        assert_eq!(fit.step(600.0, 190.0).0, Some(690.0));
        for _ in 0..PATIENCE {
            assert!(fit.step(600.0, 190.0).1 > 0.0);
        }
        assert_eq!(fit.step(600.0, 190.0), (None, 0.0));
        assert!(!fit.waiting());
    }
}
