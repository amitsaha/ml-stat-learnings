Presentation outline


1. Load sample data (non-biased)

- Talk through the data
- What the different columns mean - score is the output of a binary classifier
- Subgroups in the data
- Probability threshold can be played with, but will be fixed at 0.5 for the demos

Overall problem statement - 

is the classifier's behavior in predicting positives (as measured by recall) statistically the same across the different subgroups?

Null hypothesis - H0 -> No measurable difference in behavior across subgroups, i.e. group membership and outcome are not related

if p-value < 0.05 -> we reject the null hypothesis


2. Problem 1 - Is the difference in the recall (TP/(TP+FN)) statistically significantly different from each other for the gender subgroups?

Contingency matrix - 2x2
Run fisher's exact test, p-value > 0.05 - not significant.


3. Problem 2 - What about the age groups or ethnicities?

Now, we have a 3x2 and 5x2 contingency matrix respectively.

We cannot run a fisher's exact test as it is. What we do is call upon an extension of the fisher's test to KxC, described in "Note on an Exact Treatment of Contingency, Goodness of Fit and Other Problems of Significance".

Key insights:

Key problem of allocation of TP and FN to each group is a sampling without replacement discrete probability problem.

Example from the streamlit demo.

Hyper geometric distribution

Overall method (along with code)

1. Incoming contingency table
2. Run the chi square test of independency for the contingency table
3. Simulate tables drawing from hypergeometric distribution and calculate the chi square estimate for each table
4. Is the observed stat >= observed state of (2), keep count
5. Repeat 3 and 4 for number of reps
6. Calculate count of 4/number of reps -> that's the p-value

We are measuring "how unlikely (and therefore significant difference) is the observed contingency table?"


4. Demo back to the Fisher's exact test with 2x2 for MC simulation to show convergence

5. Demo with the ethnicity biased dataset - show the 0 column in FN for WHITE ethnicity

6. Final notes

1. This is a 2-sided test i.e. we don't know which group(s) are "adversely" affected and needs to be paired with other tests to estimate directionality

2. This is an exact test, i.e. it is using the actual TP/FN numbers, the random sampling is coming in the estimation process - also suitable for small numbers
 




