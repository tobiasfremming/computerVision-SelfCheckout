
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    # Use OpenCV with GUI support
    (python312Packages.opencv4.override {
      enableGtk3 = true;
      enableFfmpeg = true;
    })
    # Other dependencies
    python312Packages.numpy
    stdenv.cc.cc.lib
    # GUI dependencies
    gtk3
    glib
    pkg-config  # Corrected from pkgconfig to pkg-config
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export GI_TYPELIB_PATH=${pkgs.gtk3}/lib/girepository-1.0:$GI_TYPELIB_PATH
  '';
}
