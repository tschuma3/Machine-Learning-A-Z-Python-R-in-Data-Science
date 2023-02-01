# Machine-Learning-A-Z-Python-R-in-Data-Science
Course can be found at: https://www.udemy.com/course/machinelearning/

Bulding a Model
    -- 5 Methods of building models
        1. All-in
            -- Throw all the variables is the model
            -- Cases
                -- Prior knowledge
                -- You have to
                -- Preparing for backwards elimination
        2. Backward Elimination -->
            -- Step 1: Select a Significance Level (SL) to stay in the model (e.g. SL = 0.05)
            -- Step 2: Fill the full model with all possible predictors(X)
            -- Step 3: Consider the predictor with the highest P-value. If P > SL, go to Step 4, otherwise go th FIN
            -- Step 4: Remove the predictor
            -- Step 5: Fit model without this variable
            -- FIN: Your Model is Ready
        3. Forward Selection -->            Stepwise Regression refere to 2, 3, and 4, but mostly 4
            -- Step 1: Select a Significance Level (SL) to stay in the model (e.g. SL = 0.05)
            -- Step 2: Fit all simple regression models y ~ xn Select the one with the lowest P-values
            -- Step 3: Keep this variable and fit all possible models with one extra predictor added to the one you already have
            -- Step 4: Consider the predictor with lowest P-value. If P > SL, go to Step 4, otherwise go th FIN
            -- FIN: Your Model is Ready
        4. Bidirectional Elimination --> 
            -- Step 1: Select a Significance Level to enter and to stay in the model (e.g. SLENTER = 0.05, SLSTAY = 0.05)
            -- Step 2: Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter)
            -- Step 3: Perform ALL steps of Backward Elimination (old variables must have P < SLSTAY to stay)
            -- Step 4: No new variables can enter and no old variables can exit
            -- FIN: Your Model is Ready
        5. Score Comparison (All Possible Models)
            -- Step 1: Select a criterion of goodness of fit (e.g. Akaike Criterion)
            -- Step 2: Construct All Possible Regression Models 2^n-1 total combinations
            -- Step 3: Select the one with the best criterion
            -- FIN: Your Model is Ready

Residual (R) Squared
    -- Goodness of fit for the model (greater is better)
    -- Rule of thumb
        -- 1.0 = Perfect fit (suspicious)
        -- ~0.9 = Very good
        -- <0.7 = Not great
        -- <0.4 = Terrible
        -- <0 = Model makes no sense for this data
    -- Adjusted R Squared