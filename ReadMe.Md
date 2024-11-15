# Circle Fitting Project

This project reads GPS data, converts it to Cartesian coordinates, projects the points onto a 2D plane, and uses the RANSAC algorithm to fit a circle to the data. The inliers are then saved in XYZ format for visualization in Meshlab.

## Prerequisites

- Python 3.x
- pandas
- numpy
- matplotlib
- shapely
- scikit-learn
- Meshlab (for visualization)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/circle_fitting.git
    cd circle_fitting
    ```

2. Install the required Python packages:

    ```sh
    pip install pandas numpy matplotlib shapely scikit-learn
    ```

3. Install Meshlab:

    - **Ubuntu/Debian**:

        ```sh
        sudo apt update
        sudo apt install meshlab

## Usage

1. Place your GPS data file (`fordulokor.csv`) in the project directory.

2. Run the `main_circle_fitting.py` script:

    ```sh
    python circle_fitting.py
    ```

3. The script will read the GPS data, convert it to Cartesian coordinates, project the points onto a 2D plane, and run the RANSAC algorithm to fit a circle. The inliers will be saved to [inliers.xyz](http://_vscodecontentref_/0).

4. Open [inliers.xyz](http://_vscodecontentref_/1) in Meshlab for visualization:

    - Open Meshlab.
    - Go to `File` > `Import Mesh`.
    - Select [inliers.xyz](http://_vscodecontentref_/2) and open it.

## Example

```sh
python main_circle_fitting.py