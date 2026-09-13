{
  description = "Development environment for python-learn";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;

          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python314

            gcc
            gccNGPackages.libstdcxx
            pkg-config
            cmake
            ninja
            git

            cudatoolkit
            # cudaPackages.cuda_nvcc
            # cudaPackages.cuda_cudart
          ];

          shellHook = ''
            echo "Development shell for python-learn"

            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
                pkgs.gccNGPackages.libstdcxx
                # pkgs.cudaPackages.cuda_cudart
                pkgs.cudatoolkit
              ]
            }:/run/opengl-driver/lib:$LD_LIBRARY_PATH

            echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
            echo ""
          '';
        };
      }
    );
}
