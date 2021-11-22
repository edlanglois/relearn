extern crate proc_macro;
use proc_macro::TokenStream;

mod indexed;
mod space;

use syn::{GenericParam, Generics, TypeParamBound};

#[proc_macro_derive(Indexed)]
pub fn indexed_macro_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    indexed::impl_indexed_macro(&ast)
}

#[proc_macro_derive(Space, attributes(element))]
pub fn space_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    space::impl_space_trait_macro::<space::SpaceImpl>(ast)
}

#[proc_macro_derive(SubsetOrd)]
pub fn subset_ord_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    space::impl_space_trait_macro::<space::SubsetOrdImpl>(ast)
}

#[proc_macro_derive(FiniteSpace)]
pub fn finite_space_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    space::impl_space_trait_macro::<space::FiniteSpaceImpl>(ast)
}

#[proc_macro_derive(SampleSpace)]
pub fn sample_space_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    space::impl_space_trait_macro::<space::SampleSpaceImpl>(ast)
}

#[proc_macro_derive(NumFeatures)]
pub fn num_features_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    space::impl_space_trait_macro::<space::NumFeaturesImpl>(ast)
}

#[proc_macro_derive(EncoderFeatureSpace)]
pub fn encoder_feature_space_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    space::impl_space_trait_macro::<space::EncoderFeatureSpaceImpl>(ast)
}

fn add_trait_bounds(mut generics: Generics, bound: &TypeParamBound) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(ref mut type_param) = *param {
            type_param.bounds.push(bound.clone())
        }
    }
    generics
}
