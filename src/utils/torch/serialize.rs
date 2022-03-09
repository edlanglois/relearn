//! Tensor serialization and deserialization
#![allow(clippy::use_self)] // created by serde derive for KindDef

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{serde_as, Bytes, DeserializeAs, SerializeAs};
use std::borrow::Cow;
use tch::{Kind, Tensor};

/// Remote serialization definition for [`tch::Kind`].
///
/// Use `#[serde(with = "KindDef")]` when serializing a field of type [`Kind`].
#[derive(Serialize, Deserialize)]
#[serde(remote = "Kind")]
pub enum KindDef {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
}

/// System byte order serialization.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

impl ByteOrder {
    /// Native byte order for this system
    pub const fn native() -> Self {
        if cfg!(target_endian = "big") {
            Self::BigEndian
        } else {
            Self::LittleEndian
        }
    }
}

/// Remote serialization definition for [`tch::Tensor`].
///
/// Use `#[serde(with = "TensorDef")]` when serializing a field of type [`Tensor`].
///
/// The deserialized tensor is located on CPU memory.
/// Panics on deserialization if the native byte order differs from the serialized byte order.
#[serde_as]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDef<'a> {
    #[serde(with = "KindDef")]
    pub kind: Kind,
    #[serde(borrow)]
    pub shape: Cow<'a, [i64]>,
    pub requires_grad: bool,
    pub byte_order: ByteOrder,
    #[serde_as(as = "Bytes")]
    #[serde(borrow)]
    pub data: Cow<'a, [u8]>,
}

/// Create a [`TensorDef`] by copying data from a [`Tensor`].
impl<'a> From<&'_ Tensor> for TensorDef<'a> {
    fn from(tensor: &Tensor) -> Self {
        let kind = tensor.kind();
        let shape = tensor.size();
        let num_elements: usize = shape.iter().product::<i64>().try_into().unwrap();

        // Unfortunately, it is unsafe to reference Tensor data.
        // The data can be shared between Tensors and reallocated at any time.
        // Must copy instead.
        let mut data = vec![0; num_elements * kind.elt_size_in_bytes()];
        tensor.copy_data_u8(&mut data, num_elements);

        Self {
            kind,
            shape: Cow::Owned(shape),
            requires_grad: tensor.requires_grad(),
            byte_order: ByteOrder::native(),
            data: Cow::Owned(data),
        }
    }
}

/// Create a [`Tensor`] by copying data from a [`TensorDef`].
impl<'a> From<&TensorDef<'a>> for Tensor {
    fn from(t: &TensorDef<'a>) -> Self {
        assert_eq!(
            t.byte_order,
            ByteOrder::native(),
            "data has non-native byte order"
        );
        Self::of_data_size(&t.data, &t.shape, t.kind).set_requires_grad(t.requires_grad)
    }
}

impl<'a> From<TensorDef<'a>> for Tensor {
    #[inline]
    fn from(t: TensorDef<'a>) -> Self {
        Self::from(&t)
    }
}

/// Serialize a [`Tensor`]. Use `#[serde_as(as = "TensorDef")]`.
impl<'a> SerializeAs<Tensor> for TensorDef<'a> {
    fn serialize_as<S>(source: &Tensor, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        TensorDef::from(source).serialize(serializer)
    }
}

/// Deserialize a [`Tensor`] to CPU memory. Use `#[serde_as(as = "TensorDef")]`.
impl<'de: 'a, 'a> DeserializeAs<'de, Tensor> for TensorDef<'a> {
    fn deserialize_as<D>(deserializer: D) -> Result<Tensor, D::Error>
    where
        D: Deserializer<'de>,
    {
        // TODO: Direct deserialization that avoids going through TensorDef.
        // May require the serde_state crate. I need something that can immediately deserialize the
        // bytes into a tensor, which requires knowing the tensor kind and shape. I am not sure
        // that this kind of contextual / stateful deserialization is possible with serde.
        //
        // The problem with the current approach is that some deserializers can only provide
        // transient byte arrays, not borrowed byte arrays. In that case, the data bytes have to be
        // copied twice: first into TensorDef and again into Tensor.
        //
        // A possible alternative might be to avoid copying owned data in
        // `From<TensorDef> for Tensor`. Try using `of_blob`? That might leak the data.
        let tensor_def: TensorDef = Deserialize::deserialize(deserializer)?;
        Ok(tensor_def.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{assert_tokens, Token};
    use std::ops::Deref;

    /// (De)Serializable newtype wrapper for [`Tensor`].
    #[serde_as]
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct STensor(#[serde_as(as = "TensorDef")] pub Tensor);

    impl From<Tensor> for STensor {
        fn from(tensor: Tensor) -> Self {
            STensor(tensor)
        }
    }
    impl From<STensor> for Tensor {
        fn from(stensor: STensor) -> Self {
            stensor.0
        }
    }

    impl Deref for STensor {
        type Target = Tensor;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    #[test]
    fn ser_de_tokens_0d_u32_tensor() {
        let tensor = STensor(Tensor::of_slice(&[0x12345678_i32]).reshape(&[]));

        let byte_order = ByteOrder::native();
        let bytes: &'static [u8] = match byte_order {
            ByteOrder::BigEndian => &[0x12, 0x34, 0x56, 0x78],
            ByteOrder::LittleEndian => &[0x78, 0x56, 0x34, 0x12],
        };

        let tokens = [
            Token::Struct {
                name: "TensorDef",
                len: 5,
            },
            Token::Str("kind"),
            Token::UnitVariant {
                name: "KindDef",
                variant: "Int",
            },
            Token::Str("shape"),
            Token::Seq { len: Some(0) },
            Token::SeqEnd,
            Token::Str("requires_grad"),
            Token::Bool(false),
            Token::Str("byte_order"),
            Token::UnitVariant {
                name: "ByteOrder",
                variant: match byte_order {
                    ByteOrder::BigEndian => "BigEndian",
                    ByteOrder::LittleEndian => "LittleEndian",
                },
            },
            Token::Str("data"),
            Token::BorrowedBytes(bytes),
            Token::StructEnd,
        ];
        assert_tokens(&tensor, &tokens);
    }

    #[test]
    fn ser_de_tokens_empty_f32_tensor() {
        let tensor = STensor(Tensor::of_slice::<f32>(&[]));

        let bytes: &'static [u8] = &[];

        let tokens = [
            Token::Struct {
                name: "TensorDef",
                len: 5,
            },
            Token::Str("kind"),
            Token::UnitVariant {
                name: "KindDef",
                variant: "Float",
            },
            Token::Str("shape"),
            Token::Seq { len: Some(1) },
            Token::I64(0),
            Token::SeqEnd,
            Token::Str("requires_grad"),
            Token::Bool(false),
            Token::Str("byte_order"),
            Token::UnitVariant {
                name: "ByteOrder",
                variant: match ByteOrder::native() {
                    ByteOrder::BigEndian => "BigEndian",
                    ByteOrder::LittleEndian => "LittleEndian",
                },
            },
            Token::Str("data"),
            Token::BorrowedBytes(bytes),
            Token::StructEnd,
        ];
        assert_tokens(&tensor, &tokens);
    }

    #[test]
    fn ser_de_tokens_1d_f32_tensor_requires_grad() {
        let tensor = STensor(Tensor::of_slice::<f32>(&[1.0]).set_requires_grad(true));

        let bytes: &'static [u8] = &[0, 0, 128, 63];

        let tokens = [
            Token::Struct {
                name: "TensorDef",
                len: 5,
            },
            Token::Str("kind"),
            Token::UnitVariant {
                name: "KindDef",
                variant: "Float",
            },
            Token::Str("shape"),
            Token::Seq { len: Some(1) },
            Token::I64(1),
            Token::SeqEnd,
            Token::Str("requires_grad"),
            Token::Bool(true),
            Token::Str("byte_order"),
            Token::UnitVariant {
                name: "ByteOrder",
                variant: match ByteOrder::native() {
                    ByteOrder::BigEndian => "BigEndian",
                    ByteOrder::LittleEndian => "LittleEndian",
                },
            },
            Token::Str("data"),
            Token::BorrowedBytes(bytes),
            Token::StructEnd,
        ];
        assert_tokens(&tensor, &tokens);
    }

    #[test]
    fn ser_de_tokens_2d_u8_tensor() {
        let tensor = STensor(Tensor::of_slice::<u8>(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]));

        let bytes: &'static [u8] = &[1, 2, 3, 4, 5, 6];

        let tokens = [
            Token::Struct {
                name: "TensorDef",
                len: 5,
            },
            Token::Str("kind"),
            Token::UnitVariant {
                name: "KindDef",
                variant: "Uint8",
            },
            Token::Str("shape"),
            Token::Seq { len: Some(2) },
            Token::I64(2),
            Token::I64(3),
            Token::SeqEnd,
            Token::Str("requires_grad"),
            Token::Bool(false),
            Token::Str("byte_order"),
            Token::UnitVariant {
                name: "ByteOrder",
                variant: match ByteOrder::native() {
                    ByteOrder::BigEndian => "BigEndian",
                    ByteOrder::LittleEndian => "LittleEndian",
                },
            },
            Token::Str("data"),
            Token::BorrowedBytes(bytes),
            Token::StructEnd,
        ];
        assert_tokens(&tensor, &tokens);
    }

    #[test]
    fn to_from_tensordef() {
        let t0 = Tensor::of_slice::<u8>(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let td: TensorDef = (&t0).into();
        let t1: Tensor = td.into();
        assert_eq!(t0, t1);
    }

    #[test]
    fn to_from_tensordef_transposed_stride() {
        // Check that data is preserved when the stride is abnormal
        let t0 = Tensor::of_slice::<u8>(&[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .reshape(&[3, 3])
            .tr();
        let td: TensorDef = (&t0).into();
        let t1: Tensor = td.into();
        // Stride is not necessarily preserved but data order is
        assert_eq!(t0, t1);
    }
}
