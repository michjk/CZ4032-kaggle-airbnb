# CZ4042: Airbnb New User Bookings Competition

This repository contains code for CZ4042 course. The project is chosen from Kaggle. The competition can be found here ![Airbnb New User Bookings Competition](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)

## Note:
- Gradient Boosting on train_users_2 has 5 fold validation accuracy 0.637 (1 best country) and kaggle score 0.86625 (public) & 0.87062 (private) with parameter: n_estimators = 100, learning_rate = 0.1, height_tree = 6
- Gradient Boosting on train_users_exclude_NDF has validation accuracy 0.702 with parameter: n_estimators = 200, learning_rate = 0.1, height_tree = 8
- Gradient Boosting on train_users_2_NDF_vs_non_NDF has 5 fold validation error rate 0.2954836 with parameter: n_estimators = 150, learning_rate = 0.2, height_tree = 5, min_child_weigh = 10
- Stack Classifier with Logistic Regreassion (default parameter) for classifying and NDF vs non-NDF and Gradient Boosting for non-NDF with parameter: n_estimators = 200, learning_rate = 0.1, height_tree = 8 has kaggle score 0.85282 (private) and 0.84963 (public).
