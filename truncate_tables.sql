-- Disable foreign key checks
SET FOREIGN_KEY_CHECKS = 0;

-- Truncate tables
TRUNCATE TABLE embryo_predictions.class_probabilities;
TRUNCATE TABLE embryo_predictions.predictions;

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1; 