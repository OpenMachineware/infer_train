// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::types::PyAny;

use crate::tensor::Tensor;

#[pyclass]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor<f32>,
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        PyTensor { inner: Tensor::new(data, &shape) }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    pub fn data(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }

    pub fn add(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: crate::ops::math::add::add(&self.inner, &other.inner),
        }
    }

    pub fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: crate::ops::linalg::matmul::matmul(
                &self.inner,
                &other.inner,
            ),
        }
    }

    pub fn relu(&self) -> PyTensor {
        PyTensor { inner: crate::ops::activation::relu::relu(&self.inner) }
    }

    pub fn to_numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        use numpy::PyArray1;
        let data = self.inner.data().to_vec();
        let arr = PyArray1::from_vec(py, data);
        Ok(arr.into_any().unbind())
    }

    #[staticmethod]
    pub fn from_numpy(array: &Bound<PyAny>) -> PyResult<PyTensor> {
        if let Ok(data) = array.extract::<Vec<f32>>() {
            let shape = vec![data.len()];
            return Ok(PyTensor { inner: Tensor::new(data, &shape) });
        }

        let arr = array.downcast::<PyArrayDyn<f32>>().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "Cannot convert to PyArrayDyn: {}",
                e
            ))
        })?;

        let shape = arr.shape().to_vec();

        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Cannot get slice from array: {}",
                e
            ))
        })?;

        let data = slice.to_vec();

        Ok(PyTensor { inner: Tensor::new(data, &shape) })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyTensor(shape={:?}, data={:?})",
            self.inner.shape(),
            &self.inner.data()[..self.inner.len().min(5)]
        )
    }
}
