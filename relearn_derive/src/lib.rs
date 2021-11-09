extern crate proc_macro;
use proc_macro::TokenStream;

mod indexed;

#[proc_macro_derive(Indexed)]
pub fn indexed_macro_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    indexed::impl_indexed_macro(&ast)
}
