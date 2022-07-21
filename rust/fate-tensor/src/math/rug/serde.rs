use super::BInt;
use rug::integer::Order;
use rug::Integer;
use serde::{de::Visitor, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

impl Serialize for BInt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.0.to_digits::<u8>(Order::Lsf).as_ref())
    }
}
struct BIntVisitor;
impl<'de> Visitor<'de> for BIntVisitor {
    type Value = BInt;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an integer between -2^31 and 2^31")
    }

    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(BInt(Integer::from_digits(v, Order::Lsf)))
    }
}

impl<'de> Deserialize<'de> for BInt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_bytes(BIntVisitor)
    }
}
