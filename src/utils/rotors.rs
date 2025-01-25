use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::error::Error;
use std::fmt;
use rustfft::num_traits::{Float, Zero, One};

#[derive(Debug, Clone)]
pub enum GeometricError {
    DimensionMismatch,
    NormalizationError,
    InvalidRotation,
    DecompositionError,
}

impl fmt::Display for GeometricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometricError::DimensionMismatch => write!(f, "Dimension mismatch in geometric operation"),
            GeometricError::NormalizationError => write!(f, "Failed to normalize vector/rotor"),
            GeometricError::InvalidRotation => write!(f, "Invalid rotation parameters"),
            GeometricError::DecompositionError => write!(f, "Failed to decompose geometric object"),
        }
    }
}

impl Error for GeometricError {}

pub trait Two {
    fn two() -> Self;
}

impl Two for f32 {
    #[inline]
    fn two() -> Self { 2.0 }
}

impl Two for f64 {
    #[inline]
    fn two() -> Self { 2.0 }
}

pub trait Scalar: 
    Copy + 
    Add<Output = Self> + 
    Sub<Output = Self> + 
    Mul<Output = Self> + 
    Div<Output = Self> +
    AddAssign +
    SubAssign +
    MulAssign +
    DivAssign +
    Neg<Output = Self> +
    PartialOrd +
    Float +
    Zero +
    One +
    Two +
    Debug +
    Display 
{
    #[inline]
    fn eps() -> Self {
        Self::epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self) -> bool {
        (*self - *other).abs() < Self::eps()
    }
    
    #[inline]
    fn pi() -> Self;
    
    #[inline]
    fn acos(&self) -> Self;
    
    #[inline]
    fn asin(&self) -> Self;
    
    #[inline]
    fn atan2(&self, other: Self) -> Self;
    
    #[inline]
    fn sin(&self) -> Self;
    
    #[inline]
    fn cos(&self) -> Self;
}

impl Scalar for f32 {
    #[inline]
    fn pi() -> Self { std::f32::consts::PI }
    
    #[inline]
    fn acos(&self) -> Self { f32::acos(*self) }
    
    #[inline]
    fn asin(&self) -> Self { f32::asin(*self) }
    
    #[inline]
    fn atan2(&self, other: Self) -> Self { f32::atan2(*self, other) }
    
    #[inline]
    fn sin(&self) -> Self { f32::sin(*self) }
    
    #[inline]
    fn cos(&self) -> Self { f32::cos(*self) }
}

impl Scalar for f64 {
    #[inline]
    fn pi() -> Self { std::f64::consts::PI }
    
    #[inline]
    fn acos(&self) -> Self { f64::acos(*self) }
    
    #[inline]
    fn asin(&self) -> Self { f64::asin(*self) }
    
    #[inline]
    fn atan2(&self, other: Self) -> Self { f64::atan2(*self, other) }
    
    #[inline]
    fn sin(&self) -> Self { f64::sin(*self) }
    
    #[inline]
    fn cos(&self) -> Self { f64::cos(*self) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T: Scalar, const N: usize> {
    pub components: [T; N],
}

impl<T: Scalar, const N: usize> Vector<T, N> {
    pub fn new(components: [T; N]) -> Self {
        Self { components }
    }

    pub fn zero() -> Self {
        Self { components: [T::zero(); N] }
    }

    pub fn unit(axis: usize) -> Self {
        let mut v = Self::zero();
        if axis < N {
            v.components[axis] = T::one();
        }
        v
    }

    pub fn dot(&self, other: &Self) -> T {
        self.components.iter()
            .zip(other.components.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x)
    }

    pub fn magnitude_squared(&self) -> T {
        self.dot(self)
    }

    pub fn magnitude(&self) -> T {
        self.magnitude_squared().sqrt()
    }

    pub fn scale(&self, factor: T) -> Self {
        let mut result = self.clone();
        for x in &mut result.components {
            *x = *x * factor;
        }
        result
    }

    pub fn normalize(&mut self) -> Result<(), GeometricError> {
        let mag = self.magnitude();
        if mag < T::eps() {
            return Err(GeometricError::NormalizationError);
        }
        for x in &mut self.components {
            *x = *x / mag;
        }
        Ok(())
    }

    pub fn normalized(&self) -> Result<Self, GeometricError> {
        let mut result = self.clone();
        result.normalize()?;
        Ok(result)
    }

    pub fn project(&self, onto: &Self) -> Self {
        let n = onto.dot(self) / onto.magnitude_squared();
        let mut result: Vector<T, N> = onto.clone();
        for x in &mut result.components {
            *x = *x * n;
        }
        result
    }

    pub fn reject(&self, from: &Self) -> Self {
        let proj = self.project(from);
        let mut result: Vector<T, N> = self.clone();
        for i in 0..N {
            result.components[i] -= proj.components[i];
        }
        result
    }
}

impl<T: Scalar, const N: usize> Add for Vector<T, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for i in 0..N {
            result.components[i] = result.components[i] + other.components[i];
        }
        result
    }
}

impl<'a, 'b, T: Scalar, const N: usize> Add<&'b Vector<T, N>> for &'a Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &'b Vector<T, N>) -> Vector<T, N> {
        let mut result = self.clone();
        for i in 0..N {
            result.components[i] = result.components[i] + other.components[i];
        }
        result
    }
}

impl<T: Scalar, const N: usize> Sub for Vector<T, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.clone();
        for i in 0..N {
            result.components[i] = result.components[i] - other.components[i];
        }
        result
    }
}

impl<'a, 'b, T: Scalar, const N: usize> Sub<&'b Vector<T, N>> for &'a Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, other: &'b Vector<T, N>) -> Vector<T, N> {
        let mut result = self.clone();
        for i in 0..N {
            result.components[i] = result.components[i] - other.components[i];
        }
        result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bivector<T: Scalar, const N: usize> {
    // store components in canonical order matching paper's notation
    // number of components is N(N-1)/2
    pub components: Box<[T]>,
}

impl<T: Scalar, const N: usize> Add for Bivector<T, N> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (i, &x) in other.components.iter().enumerate() {
            result.components[i] = result.components[i] + x;
        }
        result
    }
}

impl<T: Scalar, const N: usize> Sub for Bivector<T, N> {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        let mut result = self.clone();
        for (i, &x) in other.components.iter().enumerate() {
            result.components[i] = result.components[i] - x;
        }
        result
    }
}

impl<'a, 'b, T: Scalar, const N: usize> Sub<&'b Bivector<T, N>> for &'a Bivector<T, N> {
    type Output = Bivector<T, N>;

    fn sub(self, other: &'b Bivector<T, N>) -> Bivector<T, N> {
        let mut result = self.clone();
        for (i, &x) in other.components.iter().enumerate() {
            result.components[i] = result.components[i] - x;
        }
        result
    }
}

impl<T: Scalar, const N: usize> SubAssign for Bivector<T, N> {
    fn sub_assign(&mut self, other: Self) {
        for (i, &x) in other.components.iter().enumerate() {
            self.components[i] = self.components[i] - x;
        }
    }
}

impl<'a, 'b, T: Scalar, const N: usize> Add<&'b Bivector<T, N>> for &'a Bivector<T, N> {
    type Output = Bivector<T, N>;

    fn add(self, other: &'b Bivector<T, N>) -> Bivector<T, N> {
        let mut result = self.clone();
        for (i, &x) in other.components.iter().enumerate() {
            result.components[i] = result.components[i] + x;
        }
        result
    }
}

impl<T: Scalar, const N: usize> AddAssign for Bivector<T, N> {
    fn add_assign(&mut self, other: Self) {
        for (i, &x) in other.components.iter().enumerate() {
            self.components[i] = self.components[i] + x;
        }
    }
}

impl<T: Scalar, const N: usize> Bivector<T, N> {
    pub const NUM_COMPONENTS: usize = (N * (N - 1)) / 2;

    pub fn zero() -> Self {
        Self {
            components: vec![T::zero(); Self::NUM_COMPONENTS].into_boxed_slice()
        }
    }

    pub fn new(components: &[T]) -> Result<Self, GeometricError> {
        if components.len() != Self::NUM_COMPONENTS {
            return Err(GeometricError::DimensionMismatch);
        }
        Ok(Self {
            components: components.to_vec().into_boxed_slice()
        })
    }

    #[inline]
    fn get_index(i: usize, j: usize) -> usize {
        debug_assert!(i != j && i < N && j < N);
        if i < j {
            (N * i - (i * (i + 1)) / 2) + (j - i - 1)
        } else {
            (N * j - (j * (j + 1)) / 2) + (i - j - 1)
        }
    }

    pub fn get(&self, i: usize, j: usize) -> T {
        if i == j || i >= N || j >= N {
            T::zero()
        } else {
            let idx = Self::get_index(i.min(j), i.max(j));
            if i < j {
                self.components[idx]
            } else {
                -self.components[idx]
            }
        }
    }

    pub fn set(&mut self, i: usize, j: usize, value: T) {
        if i != j && i < N && j < N {
            let idx = Self::get_index(i.min(j), i.max(j));
            self.components[idx] = if i < j { value } else { -value };
        }
    }

    pub fn magnitude_squared(&self) -> T {
        self.components.iter()
            .fold(T::zero(), |acc, &x| acc + x * x)
    }

    pub fn magnitude(&self) -> T {
        self.magnitude_squared().sqrt()
    }

    pub fn negate(&mut self) {
        for x in self.components.iter_mut() {
            *x = -(*x);
        }
    }

    pub fn negated(&self) -> Self {
        let mut result = self.clone();
        result.negate();
        result
    }

    pub fn normalize(&mut self) -> Result<(), GeometricError> {
        let mag = self.magnitude();
        if mag < T::eps() {
            return Err(GeometricError::NormalizationError);
        }
        for x in self.components.iter_mut() {
            *x = *x / mag;
        }
        Ok(())
    }

    pub fn from_outer_product(a: &Vector<T, N>, b: &Vector<T, N>) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in (i+1)..N {
                let val = a.components[i] * b.components[j] - 
                         a.components[j] * b.components[i];
                result.set(i, j, val);
            }
        }
        result
    }

    /// commutator product [A,B] = (AB-BA)/2 | paper section 3
    pub fn commutator(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        
        for i in 0..N {
            for j in (i+1)..N {
                let mut sum = T::zero();
                for k in 0..N {
                    if k != i && k != j {
                        sum += self.get(i,k) * other.get(k,j) -
                              self.get(j,k) * other.get(k,i);
                    }
                }
                result.set(i, j, sum);
            }
        }
        
        result
    }

    pub fn contract_vector(&self, v: &Vector<T, N>) -> Vector<T, N> {
        let mut result: Vector<T, N> = Vector::zero();
        
        for i in 0..N {
            for j in 0..N {
                if i == j { continue; }
                result.components[i] += self.get(j,i) * v.components[j];
            }
        }
        
        result
    }

    pub fn dot(&self, other: &Self) -> T {
        self.components.iter()
            .zip(other.components.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x)
    }

    pub fn scale(&self, factor: T) -> Self {
        let mut result = self.clone();
        for x in result.components.iter_mut() {
            *x = *x * factor;
        }
        result
    }

    pub fn from_components(components: &[T]) -> Result<Self, GeometricError> {
        Self::new(components)
    }

    /// Convert bivector to Euler angles (3D only)
    /// Returns (roll, pitch, yaw) in radians
    pub fn to_euler_angles(&self) -> (T, T, T) {
        assert!(N == 3, "Euler angles only valid for 3D");
        
        // For 3D bivector, components are:
        // [0] = xy plane (yaw)
        // [1] = xz plane (pitch)
        // [2] = yz plane (roll)
        
        let roll = self.get(1, 2);   // yz plane rotation
        let pitch = self.get(2, 0);  // xz plane rotation
        let yaw = self.get(0, 1);    // xy plane rotation
        
        let roll = roll.atan2(T::one());
        let pitch = pitch.atan2(T::one()).clamp(-T::pi()/T::two(), T::pi()/T::two());
        let yaw = yaw.atan2(T::one());
        
        (roll, pitch, yaw)
    }

    /// for small angles
    pub fn to_rotor(&self) -> Rotor<T, N> {
        let angle = self.magnitude();
        
        if angle < T::eps() {
            return Rotor::identity();
        }
        
        let scalar = T::one();
        let bivector = self.scale(T::one() / T::two());
        
        Rotor::new(scalar, bivector)
    }
}

#[derive(Clone, Debug)]
pub struct Rotor<T: Scalar, const N: usize> {
    pub scalar: T,
    pub bivector: Bivector<T, N>,
}

pub type Rotor2D<T> = Rotor<T, 2>;
pub type Rotor3D<T> = Rotor<T, 3>;
pub type Rotor4D<T> = Rotor<T, 4>;

impl<T: Scalar, const N: usize> Rotor<T, N> {
    /// identity rotor == no rotation
    pub fn identity() -> Self {
        Self {
            scalar: T::one(),
            bivector: Bivector::zero(),
        }
    }

    pub fn new(scalar: T, bivector: Bivector<T, N>) -> Self {
        let mut r = Self { scalar, bivector };
        r.normalize();
        r
    }

    pub fn from_vectors(from: &Vector<T, N>, to: &Vector<T, N>) -> Result<Self, GeometricError> {
        let f = from.normalized()?;
        let t = to.normalized()?;
        
        // R = (1 + t·f + t∧f)
        let dot = t.dot(&f);
        let wedge = Bivector::from_outer_product(&t, &f);
        
        let mut r = Self {
            scalar: T::one() + dot,
            bivector: wedge,
        };
        r.normalize();
        Ok(r)
    }

    pub fn from_plane_angle(i: usize, j: usize, angle: T) -> Result<Self, GeometricError> {
        if i >= N || j >= N || i == j {
            return Err(GeometricError::InvalidRotation);
        }

        let mut r = Self::identity();
        let half_angle = angle / (T::one() + T::one());
        
        r.scalar = half_angle.cos();
        r.bivector.set(i, j, half_angle.sin());
        
        Ok(r)
    }

    /// Rotate a vector by this rotor: RvR̃
    pub fn rotate(&self, v: &Vector<T, N>) -> Vector<T, N> {
        let mut temp: Vector<T, N> = Vector::zero();
        let mut result: Vector<T, N> = Vector::zero();
        
        // R * v
        for i in 0..N {
            temp.components[i] = self.scalar * v.components[i];
            for j in 0..N {
                if i == j { continue; }
                temp.components[i] += self.bivector.get(j, i) * v.components[j];
            }
        }
        
        // * R̃
        for i in 0..N {
            result.components[i] = self.scalar * temp.components[i];
            for j in 0..N {
                if i == j { continue; }
                result.components[i] -= self.bivector.get(i, j) * temp.components[j];
            }
        }
        
        result
    }

    pub fn reverse(&self) -> Self {
        Self {
            scalar: self.scalar,
            bivector: self.bivector.negated(),
        }
    }

    pub fn normalize(&mut self) {
        let norm_squared = self.scalar * self.scalar + self.bivector.magnitude_squared();
        let norm = norm_squared.sqrt();
        
        if norm > T::eps() {
            self.scalar = self.scalar / norm;
            for x in self.bivector.components.iter_mut() {
                *x = *x / norm;
            }
        }
        
        // TODO: implem Perwass algorithm for proper rotor factorization
    }

    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::identity();
        
        // Scalar part: s1s2 - <B1B2>_0
        result.scalar = self.scalar * other.scalar;
        for i in 0..N {
            for j in (i+1)..N {
                result.scalar -= self.bivector.get(i,j) * other.bivector.get(i,j);
            }
        }
        
        // Bivector part: s1B2 + s2B1 + [B1,B2]
        let commutator = self.bivector.commutator(&other.bivector);
        
        for i in 0..N {
            for j in (i+1)..N {
                let val = self.scalar * other.bivector.get(i,j) +
                         other.scalar * self.bivector.get(i,j) +
                         commutator.get(i,j);
                         
                result.bivector.set(i, j, val);
            }
        }
        
        result.normalize();
        result
    }

    /// rotate a bivector by this rotor using the sandwiching product RBR̃
    pub fn rotate_bivector(&self, b: &Bivector<T, N>) -> Bivector<T, N> {
        let mut result = Bivector::zero();
        
        // First compute R * B
        // For a rotor R = s + B₁, and bivector B₂, the product is:
        // RB₂ = sB₂ + B₁B₂
        // where B₁B₂ = -<B₁B₂>₀ + [B₁,B₂] + B₁∧B₂
        
        // compute sB₂ part
        let mut temp: Bivector<T, N> = Bivector::zero();
        for i in 0..N {
            for j in (i+1)..N {
                temp.set(i, j, self.scalar * b.get(i, j));
            }
        }
        
        // add B₁B₂ parts:
        for i in 0..N {
            for j in (i+1)..N {
                // 2a. add commutator term [B₁,B₂]
                let mut sum = T::zero();
                for k in 0..N {
                    if k != i && k != j {
                        sum += self.bivector.get(i,k) * b.get(k,j) -
                               self.bivector.get(j,k) * b.get(k,i);
                    }
                }
                temp.set(i, j, temp.get(i,j) + sum);
                
            }
        }
        
        // compute (RB₂)R̃ = (RB₂)(-B₁ + s)
        // negative sign in -B₁ comes from reversing the bivector part
        for i in 0..N {
            for j in (i+1)..N {
                let mut sum = T::zero();
                
                for k in 0..N {
                    if k != i && k != j {
                        sum += temp.get(i,k) * self.bivector.get(k,j) -
                               temp.get(j,k) * self.bivector.get(k,i);
                    }
                }
                
                sum += temp.get(i,j) * self.scalar;
                
                result.set(i, j, sum);
            }
        }
        
        result
    }

    pub fn to_axis_angle(&self) -> (Bivector<T, N>, T) {
        let angle = T::two() * self.scalar.acos();
        
        if angle < T::eps() {
            return (Bivector::zero(), T::zero());
        }
        
        let axis = self.bivector.scale(T::one() / angle.sin());
        (axis, angle)
    }

    /// for 3D only
    pub fn to_euler_angles(&self) -> (T, T, T) {

        assert!(N == 3);
        
        let w = self.scalar;
        let x = self.bivector.get(1, 2);  // yz component
        let y = self.bivector.get(2, 0);  // zx component
        let z = self.bivector.get(0, 1);  // xy component
        
        let roll = T::two() * (w * x + y * z).atan2(T::one() - T::two() * (x * x + y * y));
        let pitch = (T::two() * (w * y - z * x)).asin();
        let yaw = T::two() * (w * z + x * y).atan2(T::one() - T::two() * (y * y + z * z));
        
        (roll, pitch, yaw)
    }
}

impl<T: Scalar, const N: usize> Mul for Rotor<T, N> {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self {
        self.compose(&rhs)
    }
}

impl<T: Scalar, const N: usize> MulAssign for Rotor<T, N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.compose(&rhs);
    }
}

#[derive(Clone, Debug)]
pub struct InertialTensor<T: Scalar, const N: usize> {
    // matrix size is (N*(N-1)/2) × (N*(N-1)/2) for bivector mapping
    pub components: Box<[T]>,
    _phantom: PhantomData<[T; N]>,
}

impl<T: Scalar, const N: usize> InertialTensor<T, N> {
    const NUM_COMPONENTS: usize = (N * (N - 1)) / 2;

    pub fn zero() -> Self {
        let size = Self::NUM_COMPONENTS;
        Self {
            components: vec![T::zero(); size * size].into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> T {
        debug_assert!(i < Self::NUM_COMPONENTS && j < Self::NUM_COMPONENTS);
        self.components[i * Self::NUM_COMPONENTS + j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: T) {
        debug_assert!(i < Self::NUM_COMPONENTS && j < Self::NUM_COMPONENTS);
        self.components[i * Self::NUM_COMPONENTS + j] = value;
    }

    /// apply tensor to bivector: I(ω)
    pub fn apply(&self, b: &Bivector<T, N>) -> Bivector<T, N> {
        let mut result = Bivector::zero();
        
        for i in 0..Self::NUM_COMPONENTS {
            let mut sum = T::zero();
            for j in 0..Self::NUM_COMPONENTS {
                sum = sum + self.get(i, j) * b.components[j];
            }
            result.components[i] = sum;
        }
        
        result
    }

    /// apply tensor to single component
    pub fn apply_component(&self, i: usize, value: &T) -> T {
        let mut sum = T::zero();
        for j in 0..Self::NUM_COMPONENTS {
            sum = sum + self.get(i, j) * *value;
        }
        sum
    }

    /// transform tensor by rotor: I' = RIR̃ | paper section 4
    pub fn transform(&self, r: &Rotor<T, N>) -> Self {
        let mut result = Self::zero();
        
        let mut transform_matrix = vec![T::zero(); Self::NUM_COMPONENTS * Self::NUM_COMPONENTS];
        
        for i in 0..N {
            for j in (i+1)..N {
                let idx = Bivector::<T, N>::get_index(i, j);
                
                let mut basis = Bivector::zero();
                basis.set(i, j, T::one());
                
                let transformed = r.rotate_bivector(&basis);
                
                for k in 0..Self::NUM_COMPONENTS {
                    transform_matrix[idx * Self::NUM_COMPONENTS + k] = transformed.components[k];
                }
            }
        }
        
        // [R]₂I[R]ᵀ₂
        for i in 0..Self::NUM_COMPONENTS {
            for j in 0..Self::NUM_COMPONENTS {
                let mut sum = T::zero();
                for k in 0..Self::NUM_COMPONENTS {
                    for l in 0..Self::NUM_COMPONENTS {
                        sum = sum + 
                            transform_matrix[i * Self::NUM_COMPONENTS + k] *
                            self.get(k, l) *
                            transform_matrix[j * Self::NUM_COMPONENTS + l];
                    }
                }
                result.set(i, j, sum);
            }
        }
        
        result
    }

    pub fn from_mesh(vertices: &[Vector<T, N>], simplices: &[[usize; N]], density: T) -> Self {
        let mut tensor = Self::zero();

        for simplex in simplices {
            let mut relative_verts = Vec::with_capacity(N);
            let v0 = &vertices[simplex[0]];
            
            for &idx in &simplex[1..] {
                let mut v = vertices[idx].clone();
                for i in 0..N {
                    v.components[i] -= v0.components[i];
                }
                relative_verts.push(v);
            }

            let mut covariance = vec![T::zero(); N * N];
            let volume = compute_simplex_volume(&relative_verts);
            
            for i in 0..N {
                for j in 0..N {
                    let mut sum = T::zero();
                    for (idx, v) in relative_verts.iter().enumerate() {
                        sum = sum + v.components[i] * v.components[j] * volume / T::from(N + 2).unwrap();
                    }
                    covariance[i * N + j] = sum;
                }
            }

            for i in 0..N {
                for j in (i+1)..N {
                    let idx = Bivector::<T, N>::get_index(i, j);
                    for k in 0..N {
                        for l in 0..N {
                            // Following paper formula for converting covariance to inertia
                            let delta_ik = if i == k { T::one() } else { T::zero() };
                            let delta_il = if i == l { T::one() } else { T::zero() };
                            let delta_jk = if j == k { T::one() } else { T::zero() };
                            let delta_jl = if j == l { T::one() } else { T::zero() };
                            
                            let contribution = 
                                delta_ik * covariance[j * N + l] +
                                delta_il * covariance[j * N + k] +
                                delta_jk * covariance[i * N + l] +
                                delta_jl * covariance[i * N + k];
                                
                            tensor.components[idx] += density * contribution;
                        }
                    }
                }
            }
        }

        tensor
    }
}

#[derive(Clone, Debug)]
pub struct RigidBody<T: Scalar, const N: usize> {
    pub state: RigidBodyState<T, N>,
    pub properties: RigidBodyProperties<T, N>,
}

impl<T: Scalar, const N: usize> RigidBody<T, N> {
    pub fn new(
        position: Vector<T, N>,
        mass: T, 
        inertia: InertialTensor<T, N>,
        inverse_inertia: InertialTensor<T, N>
    ) -> Self {
        Self {
            state: RigidBodyState {
                position,
                orientation: Rotor::identity(),
                velocity: Vector::zero(),
                angular_velocity: Bivector::zero(),
            },
            properties: RigidBodyProperties {
                mass,
                inertia,
                inverse_inertia,
            }
        }
    }

    pub fn from_mesh(
        position: Vector<T, N>,
        vertices: &[Vector<T, N>],
        simplices: &[[usize; N]], 
        density: T
    ) -> Self {
        let inertia = InertialTensor::from_mesh(vertices, simplices, density);
        
        // (TODO: proper matrix inverse)
        let inverse_inertia = InertialTensor::zero(); 
        
        let mass = density * compute_mesh_volume(vertices, simplices);
        
        Self::new(position, mass, inertia, inverse_inertia)
    }
}

fn compute_simplex_volume<T: Scalar, const N: usize>(vertices: &[Vector<T, N>]) -> T {
    if vertices.len() != N {
        return T::zero();
    }
    
    let mut matrix = vec![T::zero(); N * N];
    
    for i in 0..N {
        for j in 0..N {
            matrix[i * N + j] = vertices[i].components[j];
        }
    }
    
    // TODO: Implement proper determinant computation
    T::one() / T::from(N).unwrap()
}

fn compute_mesh_volume<T: Scalar, const N: usize>(
    vertices: &[Vector<T, N>], 
    simplices: &[[usize; N]]
) -> T {
    let mut total = T::zero();
    
    for simplex in simplices {
        let mut verts = Vec::with_capacity(N);
        for &idx in simplex {
            verts.push(vertices[idx].clone());
        }
        total = total + compute_simplex_volume(&verts);
    }
    
    total
}

#[derive(Clone, Debug)]
pub struct RigidBodyState<T: Scalar, const N: usize> {
    pub position: Vector<T, N>,
    pub orientation: Rotor<T, N>,
    pub velocity: Vector<T, N>,
    pub angular_velocity: Bivector<T, N>,
}

#[derive(Clone, Debug)]
pub struct RigidBodyProperties<T: Scalar, const N: usize> {
    pub mass: T,
    pub inertia: InertialTensor<T, N>,
    pub inverse_inertia: InertialTensor<T, N>,
}

impl<T: Scalar, const N: usize> RigidBody<T, N> {
    pub fn step(&mut self, dt: T, force: &Vector<T, N>, torque: &Bivector<T, N>) {
        // transform world-space torque to body space
        // τ_body = R̃ τ_world R
        let body_torque = self.state.orientation.reverse()
            .compose(&Rotor::new(T::one(), torque.clone()))
            .compose(&self.state.orientation)
            .bivector;

        for i in 0..N {
            self.state.velocity.components[i] += 
                force.components[i] * dt / (T::one() + T::one()) / 
                self.properties.mass;
        }

        let L = self.properties.inertia.apply(&self.state.angular_velocity);
        let gyro = self.state.angular_velocity.commutator(&L);
        let mut new_L = L.clone();
        for i in 0..Bivector::<T, N>::NUM_COMPONENTS {
            new_L.components[i] += (body_torque.components[i] - gyro.components[i]) * 
                                 dt / (T::one() + T::one());
        }
        self.state.angular_velocity = 
            self.properties.inverse_inertia.apply(&new_L);

        for i in 0..N {
            self.state.position.components[i] += 
                self.state.velocity.components[i] * dt;
        }

        let omega_mag = self.state.angular_velocity.magnitude();
        if omega_mag > T::eps() {
            let angle = omega_mag * dt;
            let half_angle = angle / (T::one() + T::one());
            
            let mut dR = Rotor::new(
                half_angle.cos(),
                self.state.angular_velocity.clone()
            );
            
            let scale = half_angle.sin() / omega_mag;
            for x in dR.bivector.components.iter_mut() {
                *x = *x * scale;
            }
            
            self.state.orientation = self.state.orientation.compose(&dR);
            self.state.orientation.normalize();
        }

        for i in 0..N {
            self.state.velocity.components[i] += 
                force.components[i] * dt / (T::one() + T::one()) / 
                self.properties.mass;
        }

        let L = self.properties.inertia.apply(&self.state.angular_velocity);
        let gyro = self.state.angular_velocity.commutator(&L);
        let mut new_L = L.clone();
        for i in 0..Bivector::<T, N>::NUM_COMPONENTS {
            new_L.components[i] += (body_torque.components[i] - gyro.components[i]) * 
                                 dt / (T::one() + T::one());
        }
        self.state.angular_velocity = 
            self.properties.inverse_inertia.apply(&new_L);
    }

    pub fn rotational_kinetic_energy(&self) -> T {
        let L = self.properties.inertia.apply(&self.state.angular_velocity);
        self.state.angular_velocity.dot(&L) / (T::one() + T::one())
    }
    
    pub fn kinetic_energy(&self) -> T {
        // translational energy
        let trans_ke = self.properties.mass * 
                      self.state.velocity.magnitude_squared() / 
                      (T::one() + T::one());

        // rotational energy (L·ω/2)
        let L = self.properties.inertia.apply(&self.state.angular_velocity);
        let rot_ke = self.state.angular_velocity.dot(&L) / 
                    (T::one() + T::one());

        trans_ke + rot_ke
    }

    pub fn angular_momentum(&self) -> Bivector<T, N> {
        let body_L = self.properties.inertia.apply(&self.state.angular_velocity);
        
        // transform to world space using similarity transformation
        // L_world = R L_body R̃
        let transformed = self.state.orientation.clone()
            .compose(&Rotor::new(T::one(), body_L))
            .compose(&self.state.orientation.reverse());
        
        transformed.bivector
    }

    pub fn apply_impulse(
        &mut self,
        impulse: &Vector<T, N>,
        point: &Vector<T, N>
    ) {
        // linear velocity change: Δv = j/m
        for i in 0..N {
            self.state.velocity.components[i] += 
                impulse.components[i] / self.properties.mass;
        }

        // get point relative to center of mass
        let mut r = point.clone();
        for i in 0..N {
            r.components[i] -= self.state.position.components[i];
        }

        // angular velocity change: Δω = I⁻¹(r∧j)
        let torque_impulse = Bivector::from_outer_product(&r, impulse);
        let inertia = self.properties.inverse_inertia
            .transform(&self.state.orientation);
        let dw = inertia.apply(&torque_impulse);
        
        // add change to angular velocity
        for i in 0..Bivector::<T, N>::NUM_COMPONENTS {
            self.state.angular_velocity.components[i] += dw.components[i];
        }
    }
}

pub struct SymplecticIntegrator<T: Scalar, const N: usize> {
    pub dt: T,
    pub substeps: usize,
}

impl<T: Scalar, const N: usize> SymplecticIntegrator<T, N> {
    pub fn new(dt: T, substeps: usize) -> Self {
        Self { dt, substeps }
    }
    pub fn step(
        &self,
        body: &mut RigidBody<T, N>,
        force: &Vector<T, N>,
        torque: &Bivector<T, N>
    ) {
        let dt = self.dt / T::from(self.substeps).unwrap();
        
        for _ in 0..self.substeps {
            body.step(dt, force, torque);
        }
    }
}

pub fn create_test_cube() -> RigidBody<f64, 3> {

    let mut inertia = InertialTensor::<f64, 3>::zero();
    let i_value = 4.0 / 6.0;  // Moment of inertia for unit cube
    
    inertia.set(0, 0, i_value);  // xy plane
    inertia.set(1, 1, i_value);  // xz plane
    inertia.set(2, 2, i_value);  // yz plane

    let mut inv_inertia = InertialTensor::<f64, 3>::zero();
    let inv_i = 1.0 / i_value;
    inv_inertia.set(0, 0, inv_i);
    inv_inertia.set(1, 1, inv_i);
    inv_inertia.set(2, 2, inv_i);

    RigidBody::new(
        Vector::new([0.0, 0.0, 0.0]),  // position
        1.0,                           // mass
        inertia,
        inv_inertia
    )
}
