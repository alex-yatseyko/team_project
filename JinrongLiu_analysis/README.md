# Team Project

## Description
Possible models
1. KNN 
2. Random Forest
3. SVM (Support Vector Machine)

Data analysis process:
1. Data cleaning: remove unuseful data, handle the missing values and identify data types; dependent and response variables
2. Pick model to fit
3. Split train and test data sets(usually 80%-20%)
3. Cross-Validation, evaluate performance

Video Link:
https://vimeo.com/965691090
Name: 
Jinrong Liu

Project Obejective

Our project focuses on analyzing Asthma based on a dataset of Patient ID, Demographic details, lifestyle factors, environmental and allergy factors, Medical history, Clinical Measurements, Symptoms, Diagnosis information, and Confidential information. The goal of the analysis is to understand how the factors affect Asthma. Therefore, we select Diagnosis as a dependent variable, and others are regarded as independent variables. 

Before working on the project, we made our team's Rules of Engagement. The details are summarized below:
1. Everyone needs to take time to participate in the meeting and discussion;
2. Everyone is free to share ideas, questions, and improvements on Google doc;
3. Transparent and clear communication among the team members;
4. Everyone needs to make a commitment to complete the team project;
5. We are always welcoming any developments and suggestions, and everyone is responsible for these developments and improvements during the working period;
5. Everyone needs to engage each allocation and be able to fully complete all allocation works.

Project process

Our team wanted to know which model or models fit the dataset, so we used different models to test the dataset. For my part, I used a Multilinear regression model. The model failed as I got the negative number of R-squared. After a discussion with team members, I realized the big mistake was because the dataset was based on classification rather than numbers. Thus, I am continuing to use the model that fits for classification. I tested the three models: Random Forest, SVM(Support Vector Machine), and KNN (K-Nearest Neighbors).

Random Forest
I used the model of Random Forest to test how the model fits the dataset. However, I  did not achieve an excellent accuracy result. The accuracy rate of the model is 91.2317%. The overall result is good, but the model has an important disadvantage. The model is imbalancing as the F1-score is 0.95, and the supported data is 454 for class 0 (male); whereas there is only 0.09 of F1-score and the supported data is 25 for class 1 (female).

SVM (Support Vector Machine)
After realizing the Random Forest Model's shortcomings, I decided to use an SVM (Support Vector Machine). The overall result is evenly poorer than the first model. The F1 Score of the model is 0.68, and the supported data is 681 for class 0. For class 1, the F1 Score is 0.09, and the supported data is 37. Additionally, the ROC AUC Score of the model is 0.5281. In other words, the model is still not as good as we expected.

K-Nearest Neighbors
Due to the limited time, I decided to try the final model- KNN, which i have leared from class. Surprisingly, the result is better than the first two models. The overall result is 94.0114%, and other factors (Mean, Standard deviation, quartiles) look good. As a result, I decided to use KNN to conduct the statistical analysis.

Project analysis

PatientID:
The PatientID ranges from 5034 to 7425, with a standard deviation suggesting a spread of IDs. All IDs are unique as the count matches the number of rows.

Age:
The age of participants ranges from 5 to 79 years, with a mean age of approximately 42 years. This indicates a middle-aged adult population with a significant range, covering children to elderly.

Gender:
Gender is coded as binary (0 and 1). The mean close to 0.49 suggests a near-even split between the two genders.

Ethnicity:
Encoded with values from 0 to 3, representing different ethnic groups. The maximum value of 3 and a mean of around 0.67 indicate a diversity of ethnic backgrounds, with potential skew towards specific groups.

Education Level:
Values range from 0 to 3, which might represent increasing levels of educational attainment. The data skews towards higher education levels, with a mean around 1.31.

BMI:
Body Mass Index (BMI) values range from 15.03 to 39.96, indicating a variety from underweight to obese categories. The mean BMI is about 27, which is classified as overweight.

Smoking:
This is likely a binary feature, with a mean close to 0.14 suggesting that a smaller proportion of the population are smokers.

Physical Activity:
Weekly hours spent in physical activity range from near 0 to about 10 hours, with a mean slightly over 5 hours. This might suggest moderate activity levels across the population.

Diet Quality:
Diet quality scores range from 0 to 10, with a mean of around 5.02, indicating a moderate diet quality on average.

Sleep Quality:
Sleep quality scores also range from about 4 to 10, with a higher mean of 7.09, suggesting generally good sleep quality among participants.

Gastroesophageal Reflux:
Appears to be a binary indicator (0 or 1), with a low mean (about 0.16), indicating that few participants suffer from this condition.

Lung Function (FEV1 and FVC):
Both Forced Expiratory Volume in 1 second (FEV1) and Forced Vital Capacity (FVC) show a wide range in values, suggesting variability in lung function among the participants. The means (about 2.55 for FEV1 and 3.74 for FVC) along with their standard deviations (about 0.86 for FEV1 and 1.30 for FVC) indicate diverse respiratory health statuses.


Project results

Demographic Diversity:
The dataset spans a wide age range from 5 to 79 years, with an average age around 42 years, indicating a diverse demographic that includes children, adults, and the elderly. The standard deviation in age suggests a broad representation across different life stages.

Gender Balance: 
Gender distribution is nearly equal with a mean very close to 0.5, implying an almost even split between the two genders represented in the dataset. This balance allows for gender-comparative analyses without major adjustments for gender disparities.

Ethnic and Educational Diversity: 
The participants come from varied ethnic backgrounds with values ranging from 0 to 3, and the dataset shows a good mix, although with a slight skew towards lower numbered ethnic groups. Education levels vary from no formal education to higher education degrees, suggesting the dataset captures a wide socioeconomic spectrum.

Health Metrics Variability:
BMI values range significantly from underweight to obese categories, with a mean indicating the average participant is at the upper edge of the normal weight range. This variability, combined with a mean physical activity of about 5 hours a week and moderate scores in diet and sleep quality, underscores the potential to explore associations between lifestyle factors and health outcomes.

Specific Health Concerns:
A small proportion of the dataset indicates issues like smoking and gastroesophageal reflux, which could be critical for targeted health interventions.

In summary, the dataset provides an adequent foundation for analyzing health and lifestyle correlations, demographic impacts on health, and the possiblity to find the best fit model in the future.


