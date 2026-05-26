-- Default rate by income type: identifies highest-risk borrower segments
SELECT 
    NAME_INCOME_TYPE,
    AVG(TARGET) AS default_rate,
    COUNT(*) AS total_customers
FROM 
    application_train
GROUP BY 
    NAME_INCOME_TYPE
ORDER BY 
    default_rate DESC;

-- Average loan amount by education: shows relationship between 
-- education level and borrowing capacity
SELECT 
    NAME_EDUCATION_TYPE,
    COUNT(*) AS customer_count,
    AVG(AMT_CREDIT) AS average_loan_amount
FROM 
    application_train
GROUP BY 
    NAME_EDUCATION_TYPE
HAVING 
    COUNT(*) > 1000
ORDER BY 
    customer_count DESC;

-- Default rate by gender: controls for credit score to isolate 
-- behavioral differences
SELECT 
    CODE_GENDER,
    AVG(EXT_SOURCE_2) AS average_ext_source_2,
    AVG(TARGET) AS default_rate,
    COUNT(*) AS total_customers
FROM 
    application_train
WHERE 
    CODE_GENDER != 'XNA'   -- excludes the rare "unknown" gender records
GROUP BY 
    CODE_GENDER
ORDER BY 
    default_rate DESC;