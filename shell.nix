{ pkgs ? import <nixpkgs> {} }: with pkgs;
mkShell {
  packages = [
    SDL2
    pkg-config
    rustc
    cargo
    rust-analyzer
  ];
}
