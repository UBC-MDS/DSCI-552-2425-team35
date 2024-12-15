Improvements:
1. Improved summary section of report to include key findings (c3326a2)
2. Made changes to summary in README to include high-level interpretation of analysis findings (368c10b)
3. 

To do:
3. It would be an added bonus if you could include a comment on how to run the script, which would be a little more convenient to understand intended usage, instead of having to refer to the README.md frequently. It could be as simple as: # Usage: (example). [add comment in docstring on how to run script - copy over from README]
4.  formatting things could be improved, such as capitalization of level 3 headers
5.  Better train / test split. Your test set seems to be too small. This is highly apparent in the final confusion matrix in your report which only contains 22 records. (I see you already mention this in the end of your report but maybe a different train/test split would help)


