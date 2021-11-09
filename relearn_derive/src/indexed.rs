use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

pub(crate) fn impl_indexed_macro(ast: &DeriveInput) -> TokenStream {
    if let Data::Enum(data_enum) = &ast.data {
        let name = &ast.ident;
        let size = data_enum.variants.len();

        let generics = &ast.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let variant_identifiers: Vec<_> = data_enum
            .variants
            .iter()
            .map(|variant| match &variant.fields {
                Fields::Unit => &variant.ident,
                _ => panic!(
                    "This derive only applies to enums whose variants have no internal data. \
                        This is violated by {}::{}",
                    name, &variant.ident
                ),
            })
            .collect();

        let index_arms = variant_identifiers
            .iter()
            .enumerate()
            .map(|(i, ident)| quote! {Self::#ident => #i});

        let from_index_arms = variant_identifiers
            .iter()
            .enumerate()
            .map(|(i, ident)| quote! {#i => Some(Self::#ident),});

        let gen = quote! {
            impl #impl_generics ::relearn::spaces::Indexed for #name #ty_generics #where_clause {
                const SIZE: usize = #size;

                fn index(&self) -> usize {
                    match *self {
                        #(#index_arms),*
                    }
                }

                fn from_index(index: usize) -> Option<Self> {
                    match index {
                        #(#from_index_arms)*
                        _ => None
                    }
                }
            }
        };
        gen.into()
    } else {
        panic!("This derive can only be applied to Enum types.");
    }
}
