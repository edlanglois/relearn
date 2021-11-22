use super::add_trait_bounds;
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, quote_spanned, ToTokens};
use std::iter::{self, Empty, Enumerate, Map};
use syn::{
    parse_quote, spanned::Spanned, AttrStyle, Attribute, Data, DeriveInput, Field, Fields,
    FieldsNamed, FieldsUnnamed, GenericParam, Generics, Ident, Index, PathArguments, Type,
    TypeTuple,
};

/// Macro that implements a trait on a struct implementing [`Space`](relearn::spaces::Space).
pub(crate) fn impl_space_trait_macro<T: SpaceTraitImpl>(input: DeriveInput) -> TokenStream {
    match &input.data {
        // Product space over the inner spaces
        // Each field is expected to be a space.
        Data::Struct(data) => match data.fields {
            Fields::Named(ref fields) => {
                T::impl_trait(input.ident, input.generics, (fields, &input.attrs as &[_]))
            }
            Fields::Unnamed(ref fields) => T::impl_trait(input.ident, input.generics, fields),
            Fields::Unit => T::impl_trait(input.ident, input.generics, ()),
        },
        _ => unimplemented!("only supports structs"),
    }
    .into()
}

/// Generic view of a structure implementing [`Space`](relearn::spaces::Space).
pub(crate) trait SpaceStruct {
    type FieldId: ToTokens;
    type FieldType: ToTokens;
    type FieldIter: Iterator<Item = (Self::FieldId, Self::FieldType, Span)>
        + DoubleEndedIterator
        + ExactSizeIterator
        + Clone;
    type ElementType: ToTokens;

    /// Iterator over (id, type, span) for each field, where `self.#id` is the field value.
    fn fields(&self) -> Self::FieldIter;

    /// The associated element type.
    fn element_type(&self) -> Self::ElementType;

    /// Construct a new element given a value for each field (in order).
    fn new_element<I>(&self, values: I) -> TokenStream2
    where
        I: Iterator,
        I::Item: ToTokens;
}

#[allow(clippy::type_complexity)]
impl<'a> SpaceStruct for (&'a FieldsNamed, &'a [Attribute]) {
    type FieldId = &'a Ident;
    type FieldType = &'a Type;
    type FieldIter = Map<syn::punctuated::Iter<'a, Field>, fn(&Field) -> (&Ident, &Type, Span)>;
    type ElementType = Type;

    fn fields(&self) -> Self::FieldIter {
        self.0
            .named
            .iter()
            .map(|f| (f.ident.as_ref().unwrap(), &f.ty, f.span()))
    }

    fn element_type(&self) -> Self::ElementType {
        self.1
            .iter()
            .find(|a| matches!(a.style, AttrStyle::Outer) && a.path.is_ident("element"))
            .expect("must specify #[element(ElementType)] attribute")
            .parse_args()
            .unwrap()
    }

    fn new_element<I>(&self, values: I) -> TokenStream2
    where
        I: Iterator,
        I::Item: ToTokens,
    {
        let element_name = into_type_name(self.element_type());
        let field_name = self.0.named.iter().map(|f| &f.ident);
        quote! {
            // Note: Using a qualified path here is not yet supported
            // https://github.com/rust-lang/rust/issues/86935
            // otherwise this could be <Self as ::relearn::spaces::Space>::Element { ... }
            #element_name {
                #( #field_name: #values, )*
            }
        }
    }
}

/// The name of a type without generics
fn into_type_name(mut ty: Type) -> Type {
    if let Type::Path(ref mut path_type) = &mut ty {
        if let Some(segment) = path_type.path.segments.last_mut() {
            if matches!(segment.arguments, PathArguments::AngleBracketed(_)) {
                segment.arguments = PathArguments::None;
            }
        }
    }
    ty
}

#[allow(clippy::type_complexity)]
impl<'a> SpaceStruct for &'a FieldsUnnamed {
    type FieldId = Index;
    type FieldType = &'a Type;
    type FieldIter = Map<
        Enumerate<syn::punctuated::Iter<'a, Field>>,
        fn((usize, &Field)) -> (Index, &Type, Span),
    >;
    type ElementType = TokenStream2;

    fn fields(&self) -> Self::FieldIter {
        self.unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| (Index::from(i), &f.ty, f.span()))
    }

    fn element_type(&self) -> Self::ElementType {
        let field_elements = self.unnamed.iter().map(|f| {
            let ty = &f.ty;
            quote_spanned! {f.span()=>
                <#ty as ::relearn::spaces::Space>::Element
            }
        });
        // The trailing comma is important so that the one-field case is still a tuple
        quote! { ( #( #field_elements, )* ) }
    }

    fn new_element<I>(&self, values: I) -> TokenStream2
    where
        I: Iterator,
        I::Item: ToTokens,
    {
        // The trailing comma is important so that the one-field case is still a tuple
        quote! { ( #( #values, )* ) }
    }
}

impl SpaceStruct for () {
    /// Arbitrary value. No fields are produced.
    type FieldId = u8;
    type FieldType = Type;
    type FieldIter = Empty<(Self::FieldId, Self::FieldType, Span)>;
    type ElementType = TypeTuple;

    fn fields(&self) -> Self::FieldIter {
        iter::empty()
    }

    fn element_type(&self) -> Self::ElementType {
        parse_quote! {()}
    }

    fn new_element<I>(&self, values: I) -> TokenStream2
    where
        I: Iterator,
        I::Item: ToTokens,
    {
        assert_eq!(values.count(), 0);
        quote! {()}
    }
}

/// Implements a trait on a space
pub(crate) trait SpaceTraitImpl {
    /// Implement a trait for a struct implementing [`Space`](relearn::spaces::Space).
    ///
    /// # Args
    /// * `name`     - Name of the struct
    /// * `generics` - Generics in the struct definition
    /// * `struct_`  - Space struct fields / element.
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, struct_: T) -> TokenStream2;
}

pub(crate) struct SpaceImpl;
impl SpaceTraitImpl for SpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::Space));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let element_type = struct_.element_type();
        let field_contains = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::Space::contains(&self.#id, &value.#id)
            }
        });
        let contains = quote! {
            true #(&& #field_contains)*
        };

        quote! {
            impl #impl_generics ::relearn::spaces::Space for #name #ty_generics #where_clause {
                type Element = #element_type;

                fn contains(&self, value: &Self::Element) -> bool {
                    #contains
                }
            }
        }
    }
}

pub(crate) struct SubsetOrdImpl;
impl SpaceTraitImpl for SubsetOrdImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::SubsetOrd));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let field_cmp = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::SubsetOrd::subset_cmp(&self.#id, &other.#id)
            }
        });
        quote! {
            impl #impl_generics ::relearn::spaces::SubsetOrd for #name #ty_generics #where_clause {
                fn subset_cmp(&self, other: &Self) -> Option<::std::cmp::Ordering> {
                    let mut cmp = ::std::cmp::Ordering::Equal;
                    #( cmp = ::relearn::spaces::product_subset_ord(cmp, #field_cmp)?; )*
                    Some(cmp)
                }
            }
        }
    }
}

pub(crate) struct FiniteSpaceImpl;
impl SpaceTraitImpl for FiniteSpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::FiniteSpace));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let field_size = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::FiniteSpace::size(&self.#id)
            }
        });
        let field_size_rev = field_size.clone().rev();

        let field_to_index_rev = struct_.fields().rev().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::FiniteSpace::to_index(&self.#id, &element.#id)
            }
        });

        let field_from_index = struct_.fields().map(|(id, _, span)| {
            let size = quote_spanned! {span=>
                ::relearn::spaces::FiniteSpace::size(&self.#id)
            };
            let from_index = quote_spanned! {span=>
                ::relearn::spaces::FiniteSpace::from_index(&self.#id, field_index)
            };
            quote! {
                {
                    let size = #size;
                    let field_index = index % size;
                    index /= size;
                    #from_index?
                }
            }
        });
        let element_from_index = struct_.new_element(field_from_index);

        quote! {
            impl #impl_generics ::relearn::spaces::FiniteSpace for #name #ty_generics #where_clause {
                fn size(&self) -> usize {
                    let mut size: usize = 1;
                    #( size = size.checked_mul(#field_size).expect("size overflows usize"); )*
                    size
                }

                fn to_index(&self, element: &Self::Element) -> usize {
                    // Little-endian number in terms of the inner element indices
                    let mut index = 0;
                    #(
                        index *= #field_size_rev;
                        index += #field_to_index_rev;
                    )*
                    index
                }

                // Relies on struct field values being evaluated in order (as written, but init
                // order works too) in the struct constructor.
                #[allow(clippy::eval_order_dependence)]
                fn from_index(&self, mut index: usize) -> Option<Self::Element> {
                    let element = #element_from_index;
                    if index == 0 { Some(element) } else { None }
                }
            }
        }
    }
}

pub(crate) struct SampleSpaceImpl;
impl SpaceTraitImpl for SampleSpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, mut generics: Generics, struct_: T) -> TokenStream2 {
        // Add distribution trait bounds
        for param in &mut generics.params {
            if let GenericParam::Type(ref mut type_param) = *param {
                let ident = &type_param.ident;
                let span = type_param.span();
                type_param.bounds.push(
                    syn::parse2(quote_spanned! {span=>
                        ::relearn::spaces::Space
                    })
                    .unwrap(),
                );
                type_param.bounds.push(syn::parse2(quote_spanned!{span=>
                    ::rand::distributions::Distribution<<#ident as ::relearn::spaces::Space>::Element>
                }).unwrap());
            }
        }
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let sampled_element = struct_.new_element(struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::rand::distributions::Distribution::sample(&self.#id, rng)
            }
        }));

        quote! {
            impl #impl_generics ::rand::distributions::Distribution<<Self as ::relearn::spaces::Space>::Element>
                for #name #ty_generics #where_clause {

                #[allow(clippy::unused_unit)]
                fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> <Self as ::relearn::spaces::Space>::Element {
                    #sampled_element
                }
            }
        }
    }
}

pub(crate) struct NumFeaturesImpl;
impl SpaceTraitImpl for NumFeaturesImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::NumFeatures));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let field_num_features = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::NumFeatures::num_features(&self.#id)
            }
        });

        quote! {
            impl #impl_generics ::relearn::spaces::NumFeatures for #name #ty_generics #where_clause {
                fn num_features(&self) -> usize {
                    0 #( + #field_num_features )*
                }
            }
        }
    }
}

pub(crate) struct EncoderFeatureSpaceImpl;
impl SpaceTraitImpl for EncoderFeatureSpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(
            generics,
            &parse_quote!(::relearn::spaces::EncoderFeatureSpace),
        );
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let field_encoder_type = struct_.fields().map(|(_, ty, span)| {
            quote_spanned! {span=>
                <#ty as ::relearn::spaces::EncoderFeatureSpace>::Encoder
            }
        });
        let field_encoder = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::EncoderFeatureSpace::encoder(&self.#id)
            }
        });
        let field_num_features = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::NumFeatures::num_features(&self.#id)
            }
        });

        let field_encoder_features_out = struct_.fields().enumerate().map(|(i, (id, _, span))| {
            let idx = Index::from(i);
            let range = if i == 0 {
                quote! {..encoder.1[#i]}
            } else {
                quote! {encoder.1[#i-1]..encoder.1[#i]}
            };
            quote_spanned! {span=>
                ::relearn::spaces::EncoderFeatureSpace::encoder_features_out(
                    &self.#id,
                    &element.#id,
                    &mut out[#range],
                    zeroed,
                    &encoder.0.#idx
                )
            }
        });
        let num_fields = struct_.fields().len();

        quote! {
            impl #impl_generics ::relearn::spaces::EncoderFeatureSpace for #name #ty_generics #where_clause {
                /// Encoder type
                ///
                /// A tuple ( field_encoders: (...), feature_ends: [usize; num_fields] )
                /// where `feature_ends[i-1]..feature_ends[i]` is range of features corresponding
                /// to field `i`.
                type Encoder = (
                    ( #(#field_encoder_type,)* ),
                    [usize; #num_fields]
                );

                // Relies on array values being evaluated in order.
                #[allow(clippy::eval_order_dependence)]
                fn encoder(&self) -> Self::Encoder {
                    // The trailing comma is important so that the one-field case is still a tuple
                    let field_encoders = ( #(#field_encoder,)* );
                    let mut offset = 0;
                    let feature_ends = [ #( {
                        offset += #field_num_features;
                        offset
                    } ),* ];
                    (field_encoders, feature_ends)
                }

                fn encoder_features_out<F: ::num_traits::Float>(
                    &self,
                    element: &Self::Element,
                    out: &mut [F],
                    zeroed: bool,
                    encoder: &Self::Encoder,
                ) {
                    #( #field_encoder_features_out; )*
                }
            }
        }
    }
}

pub(crate) struct LogElementSpaceImpl;
impl SpaceTraitImpl for LogElementSpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: Ident, generics: Generics, _struct: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::Space));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        quote! {
            impl #impl_generics ::relearn::spaces::ElementRefInto<::relearn::logging::Loggable> for #name #ty_generics #where_clause {
                fn elem_ref_into(&self, _element: &Self::Element) -> ::relearn::logging::Loggable {
                    ::relearn::logging::Loggable::Nothing
                }
            }
        }
    }
}
