{
  description = "Python environment with PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Change to aarch64-linux, x86_64-darwin, or aarch64-darwin if needed
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          (pkgs.python3.withPackages (
            python-pkgs: with python-pkgs; [
              torch
            ]
          ))
        ];

        shellHook = ''
          echo "Python 3 Flake environment activated!"
          echo "PyTorch is ready to use."
          python --version
        '';
      };
    };
}
