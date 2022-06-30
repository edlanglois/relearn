use super::add_trait_bounds;
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, quote_spanned, ToTokens};
use std::iter::{self, Empty, Enumerate, Map};
use syn::{
    parse_quote, spanned::Spanned, AttrStyle, Attribute, Data, DeriveInput, Field, Fields,
    FieldsNamed, FieldsUnnamed, GenericParam, Generics, Ident, Index, PathArguments, Type,
};

/// Macro that implements a trait on a struct implementing [`Space`](relearn::spaces::Space).
pub(crate) fn impl_space_trait_macro<T: SpaceTraitImpl>(input: DeriveInput) -> TokenStream {
    // Optional associated element type set explicitly with #[element(ElementType)]
    // Required for NamedSpacedStruct, optional for Unnamed.
    let element_type = get_element_type(&input.attrs);
    match &input.data {
        // Product space over the inner spaces
        // Each field is expected to be a space.
        Data::Struct(data) => match data.fields {
            Fields::Named(ref fields) => T::impl_trait(
                &input.ident,
                input.generics,
                NamedSpaceStruct::new(
                    fields,
                    element_type.expect("must specify #[element(ElementType)] attribute"),
                ),
            ),
            Fields::Unnamed(ref fields) => T::impl_trait(
                &input.ident,
                input.generics,
                UnnamedSpaceStruct::new(fields, element_type),
            ),
            Fields::Unit => T::impl_trait(
                &input.ident,
                input.generics,
                UnitSpaceStruct::new(element_type),
            ),
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

fn get_element_type(attributes: &[Attribute]) -> Option<Type> {
    Some(
        attributes
            .iter()
            .find(|a| matches!(a.style, AttrStyle::Outer) && a.path.is_ident("element"))?
            .parse_args()
            .expect("error parsing #[element(ElementType)]"),
    )
}

/// A [`SpaceStruct`] with named fields
#[derive(Clone)]
struct NamedSpaceStruct<'a> {
    fields: &'a FieldsNamed,
    element_type: Type,
}

impl<'a> NamedSpaceStruct<'a> {
    pub fn new(fields: &'a FieldsNamed, element_type: Type) -> Self {
        Self {
            fields,
            element_type,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<'a> SpaceStruct for NamedSpaceStruct<'a> {
    type FieldId = &'a Ident;
    type FieldType = &'a Type;
    type FieldIter = Map<syn::punctuated::Iter<'a, Field>, fn(&Field) -> (&Ident, &Type, Span)>;
    type ElementType = Type;

    fn fields(&self) -> Self::FieldIter {
        self.fields
            .named
            .iter()
            .map(|f| (f.ident.as_ref().unwrap(), &f.ty, f.span()))
    }

    fn element_type(&self) -> Self::ElementType {
        self.element_type.clone()
    }

    fn new_element<I>(&self, values: I) -> TokenStream2
    where
        I: Iterator,
        I::Item: ToTokens,
    {
        let element_name = into_type_name(self.element_type());
        let field_name = self.fields.named.iter().map(|f| &f.ident);
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

/// A [`SpaceStruct`] with unnamed fields
#[derive(Clone)]
struct UnnamedSpaceStruct<'a> {
    fields: &'a FieldsUnnamed,
    element_type: Option<Type>,
}

impl<'a> UnnamedSpaceStruct<'a> {
    pub fn new(fields: &'a FieldsUnnamed, element_type: Option<Type>) -> Self {
        Self {
            fields,
            element_type,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<'a> SpaceStruct for UnnamedSpaceStruct<'a> {
    type FieldId = Index;
    type FieldType = &'a Type;
    type FieldIter = Map<
        Enumerate<syn::punctuated::Iter<'a, Field>>,
        fn((usize, &Field)) -> (Index, &Type, Span),
    >;
    type ElementType = TokenStream2;

    fn fields(&self) -> Self::FieldIter {
        self.fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| (Index::from(i), &f.ty, f.span()))
    }

    fn element_type(&self) -> Self::ElementType {
        // Return the element type if set
        if let Some(element_type) = self.element_type.as_ref().cloned() {
            return quote! { #element_type };
        }

        // Otherwise return an anonymous tuple
        let field_elements = self.fields.unnamed.iter().map(|f| {
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
        if let Some(element_type) = self.element_type.as_ref().cloned() {
            quote! { #element_type ( #( #values ),* ) }
        } else {
            // The trailing comma is important so that the one-field case is still a tuple
            quote! { ( #( #values, )* ) }
        }
    }
}

/// A [`SpaceStruct`] with no fields
#[derive(Clone)]
struct UnitSpaceStruct {
    element_type: Option<Type>,
}

impl UnitSpaceStruct {
    pub fn new(element_type: Option<Type>) -> Self {
        Self { element_type }
    }
}

impl SpaceStruct for UnitSpaceStruct {
    /// Arbitrary value. No fields are produced.
    type FieldId = u8;
    type FieldType = Type;
    type FieldIter = Empty<(Self::FieldId, Self::FieldType, Span)>;
    type ElementType = TokenStream2;

    fn fields(&self) -> Self::FieldIter {
        iter::empty()
    }

    fn element_type(&self) -> Self::ElementType {
        // The element type if set otherwise ()
        if let Some(element_type) = self.element_type.as_ref().cloned() {
            quote! { #element_type }
        } else {
            quote! { () }
        }
    }

    fn new_element<I>(&self, values: I) -> TokenStream2
    where
        I: Iterator,
        I::Item: ToTokens,
    {
        assert_eq!(values.count(), 0);
        if let Some(element_type) = self.element_type.as_ref().cloned() {
            quote! { #element_type }
        } else {
            quote! { () }
        }
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
    fn impl_trait<T: SpaceStruct + Clone>(
        name: &Ident,
        generics: Generics,
        struct_: T,
    ) -> TokenStream2;
}

pub(crate) struct SpaceImpl;
impl SpaceTraitImpl for SpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2 {
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
    fn impl_trait<T: SpaceStruct>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2 {
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
    fn impl_trait<T: SpaceStruct>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2 {
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

pub(crate) struct NonEmptySpaceImpl;
impl SpaceTraitImpl for NonEmptySpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::NonEmptySpace));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let some_element = struct_.new_element(struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::NonEmptySpace::some_element(&self.#id)
            }
        }));

        quote! {
            #[allow(clippy::unused_unit)]
            impl #impl_generics ::relearn::spaces::NonEmptySpace for #name #ty_generics #where_clause {
                fn some_element(&self) -> <Self as ::relearn::spaces::Space>::Element {
                    #some_element
                }
            }
        }
    }
}

pub(crate) struct SampleSpaceImpl;
impl SpaceTraitImpl for SampleSpaceImpl {
    fn impl_trait<T: SpaceStruct>(
        name: &Ident,
        mut generics: Generics,
        struct_: T,
    ) -> TokenStream2 {
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

pub(crate) struct FeatureSpaceImpl;
impl SpaceTraitImpl for FeatureSpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics = add_trait_bounds(generics, &parse_quote!(::relearn::spaces::FeatureSpace));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let field_num_features = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                ::relearn::spaces::FeatureSpace::num_features(&self.#id)
            }
        });

        let field_features_out = struct_.fields().map(|(id, _, span)| {
            quote_spanned! {span=>
                out = ::relearn::spaces::FeatureSpace::features_out(
                    &self.#id,
                    &element.#id,
                    out,
                    zeroed);
            }
        });

        let num_fields = struct_.fields().len();

        let option_batch_features_out = if num_fields == 0 {
            // Custom implementation when there are no fields to avoid iterating over elements
            Some(quote! {
                #[inline]
                fn batch_features_out<'a, I, A>(
                    &self,
                    elements: I,
                    out: &mut ::ndarray::ArrayBase<A, ::ndarray::Ix2>,
                    zeroed: bool
                ) where
                    I: IntoIterator<Item = &'a Self::Element>,
                    Self::Element: 'a,
                    A: ::ndarray::DataMut,
                    A::Elem: ::num_traits::Float,
                {
                }
            })
        } else {
            None
        };

        quote! {
            impl #impl_generics ::relearn::spaces::FeatureSpace for #name #ty_generics #where_clause {
                #[inline]
                fn num_features(&self) -> usize {
                    0 #( + #field_num_features )*
                }

                #[inline]
                fn features_out<'a, F: ::num_traits::Float>(
                    &self,
                    element: &Self::Element,
                    mut out: &'a mut [F],
                    zeroed: bool,
                ) -> &'a mut [F] {
                    #( #field_features_out; )*
                    out
                }

                #option_batch_features_out
            }
        }
    }
}

pub(crate) struct LogElementSpaceImpl;
impl SpaceTraitImpl for LogElementSpaceImpl {
    fn impl_trait<T: SpaceStruct>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2 {
        let generics =
            add_trait_bounds(generics, &parse_quote!(::relearn::spaces::LogElementSpace));
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let field_log = struct_.fields().map(|(id, _, span)| {
            let id_name = format!("{}", id.to_token_stream());
            quote_spanned! {span=>
                ::relearn::spaces::LogElementSpace::log_element(&self.#id, #id_name, &element.#id, &mut logger)
            }
        });
        quote! {
            impl #impl_generics ::relearn::spaces::LogElementSpace for #name #ty_generics #where_clause {
                #[inline]
                fn log_element<L: ::relearn::logging::StatsLogger + ?Sized>(
                    &self,
                    name: &'static str,
                    element: &Self::Element,
                    logger: &mut L,
                ) -> Result<(), ::relearn::logging::LogError> {
                    let mut logger = ::relearn::logging::StatsLogger::group(
                        ::relearn::logging::StatsLogger::with_scope(logger, name));
                    #( #field_log?; )*
                    Ok(())
                }
            }
        }
    }
}

/// Derives [`Space`] and all other common space traits for a struct as a product space.
pub(crate) struct ProductSpaceImpl;
impl SpaceTraitImpl for ProductSpaceImpl {
    fn impl_trait<T>(name: &Ident, generics: Generics, struct_: T) -> TokenStream2
    where
        T: SpaceStruct + Clone,
    {
        let impls = [
            SpaceImpl::impl_trait(name, generics.clone(), struct_.clone()),
            SubsetOrdImpl::impl_trait(name, generics.clone(), struct_.clone()),
            NonEmptySpaceImpl::impl_trait(name, generics.clone(), struct_.clone()),
            SampleSpaceImpl::impl_trait(name, generics.clone(), struct_.clone()),
            FeatureSpaceImpl::impl_trait(name, generics.clone(), struct_.clone()),
            LogElementSpaceImpl::impl_trait(name, generics, struct_),
        ];

        impls.into_iter().collect()
    }
}
