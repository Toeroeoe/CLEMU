# What do we need to do?
- A generalizable, yearly emulator for 
    - carbon pools and fluxes in CLM5
    - nitrogen pools and fluxes in CLM5
    - Adjacent variables that are technically required for smooth initialization

# Predictors:
    - Yearly climate data (including seasonal variation?)
    - Parameters (large selection?)
    - surface static variables (large selection?)
    - Other inputs? e.g., ndep, CO2

# The model we require
    - Graph neural network
        - Here, we can embeedd graph level, node level, and link level features
        - This is nice because some "global" featuers such as land cover type can be embedded as graph level features
        - We can also embed other graph level features such as soil texture, soil depth, etc.
        - Node level features can be things like soil moisture, mean and std seasonal temperature, etc.
        - But what would be the link level features?
        - What type of features do we extract from the variables?
        - And does the model do that automatically? Or i have to do that at a certain stage?


# The holistic tool
    - Once fully developed, the whole tool should take:
        - the parameter file
        - the surface file
        - the forcing input (not sure yet how)
        - a template restart file one time step after coldstart