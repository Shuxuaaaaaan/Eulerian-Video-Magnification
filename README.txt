███████╗██╗   ██╗███╗   ███╗
██╔════╝██║   ██║████╗ ████║
█████╗  ██║   ██║██╔████╔██║
██╔══╝  ╚██╗ ██╔╝██║╚██╔╝██║
███████╗ ╚████╔╝ ██║ ╚═╝ ██║
╚══════╝  ╚═══╝  ╚═╝     ╚═╝


DESCRIPTION
    Eulerian Video Magnification for colors and motions magnification

USAGE
    python evm.py [-h] --video_path VIDEO_PATH [--level LEVEL] [--alpha ALPHA]
                  [--lambda_cutoff LAMBDA_CUTOFF] [--low_omega LOW_OMEGA]
                  [--high_omega HIGH_OMEGA] --saving_path SAVING_PATH
                  [--mode {gaussian,laplacian}] [--attenuation ATTENUATION]

    arguments:
      --video_path VIDEO_PATH, -v VIDEO_PATH
                            Path to the video to be used
      --level LEVEL, -l LEVEL
                            Number of level of the Gaussian/Laplacian Pyramid
      --alpha ALPHA, -a ALPHA
                            Amplification factor
      --lambda_cutoff LAMBDA_CUTOFF, -lc LAMBDA_CUTOFF
                            λ cutoff for Laplacian EVM
      --low_omega LOW_OMEGA, -lo LOW_OMEGA
                            Minimum allowed frequency
      --high_omega HIGH_OMEGA, -ho HIGH_OMEGA
                            Maximum allowed frequency
      --saving_path SAVING_PATH, -s SAVING_PATH
                            Saving path of the magnified video (.avi extension necessary)
      --mode {gaussian,laplacian}, -m {gaussian,laplacian}
                            Type of pyramids to use (gaussian or laplacian)
      --attenuation ATTENUATION, -at ATTENUATION
                            Attenuation factor for I and Q channel post filtering

REFERENCES
    Eulerian Video Magnification for Revealing Subtle Changes in the World (https://people.csail.mit.edu/mrub/evm/)

CONTRIBUTORS
    Hussem Ben Belgacem

Configurations to obtain the same results

Colors
    - face
        - level: 4
        - alpha: 50
        - attenuation: 1
        - mode: gaussian
        - low omega: 0.8333
        - high omega: 1
    - face2
        - level: 6
        - alpha: 50
        - attenuation: 1
        - mode: gaussian
        - low omega: 0.8333
        - high omega: 1

Motions
    - wrist
        - level: 6
        - alpha: 30
        - lambda cutoff: 16
        - attenuation: 0.1
        - mode: laplacian
        - low omega: 0.4
        - high omega: 3

    - baby
        - level: 4
        - alpha: 15
        - lambda cutoff: 16
        - attenuation: 0.1
        - mode: laplacian
        - low omega: 0.4
        - high omega: 3