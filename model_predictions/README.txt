This is a google Colab python based tool that analyzes the Google Co-Authorship Network. 
This tool provides a framework for analyzing this network which is derived for the real-world datasets. 
The tool mainly focuses on the predictive models to explore author relationships, centrality, and collaboration patterns. 
It provides visualizations, statistical analysis, and predictive models for various author related attributes.

In order to use this tool, you need to download the three network datasets:
1. authorsfeatures.csv
2. authorsFields.csv
3. coauthorship.csv
Since the dataset is quite large, uploading the entire dataset can take a long time. You can download the datasets
by following the link in this orignal github repo: https://github.com/kalhorghazal/Google-Scholar-Paper?tab=readme-ov-file
And upload them to your own google drive in a single file. Then you need to update the path of 
the three files accordingly to your google drive file path in order to use this google colab script

Since the script is built on top of the google colab environment, you will not need to make any additional changes
after downloading the three network datasets and upload to your google drive. Once the dataset is uploaded and you 
have changed these file paths (which is indicated in the code section as well), you may run the tool on google colab
