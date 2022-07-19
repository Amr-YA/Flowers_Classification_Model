# The Command Line Application
Converting the trained model into an application that others can use. Application is a Python script that can run from the command line. The predict.py module will predict the top flower names from an image along with their corresponding probabilities.

## Usage:
`python predict.py /path/to/image saved_model`
* The path to the image is provided as input argument
* The path to the trained model is provided as input argument, if not provided or not found a default model will be used instead

Option 1:
`python predict.py /path/to/image saved_model --top_k n`
* --top_k : Return the top n most likely classes

`python predict.py /path/to/image saved_model --category_names map.json`
* --category_names : Path to a JSON file mapping labels to flower names, if not found a default mapping will be used