import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import tensorflow as tf
import warnings


def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def rmse_loss (ori_data, imputed_data, data_m):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
    
  # Only for missing values
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  denominator = np.sum(1-data_m)
  
  rmse = np.sqrt(nominator/float(denominator))
  
  return rmse


def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.math.sqrt(in_dim / 2.)
  return tf.random.normal(shape = size, stddev = xavier_stddev)
      

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx

def MCAR2unifo(inputX, mr):
    """
    Generate MCAR (Missing Completely At Random) data by introducing missing values 
    into the input matrix at a given missing rate.

    Parameters:
        inputX (ndarray): Matrix of data (n patterns x p features)
        mr (float): Desired missing rate (percentage, 0-100)

    Returns:
        ndarray: Matrix with missing values introduced (NaN)

    References:
        Garciarena, Unai, and Roberto Santana. "An extensive analysis of the interaction between missing data types, 
        imputation methods, and supervised classifiers." Expert Systems with Applications 89 (2017): 52-65.
    """
    outputX = inputX.copy()

    # Determine dimensions
    n, p = outputX.shape

    # Calculate total number of elements to make missing
    T = int(round(n * p * mr / 100))

    # Randomly select T unique indices from the flattened array
    indices = np.random.choice(n * p, T, replace=False)

    # Flatten the array for easier indexing
    flat_array = outputX.ravel()
    
    # Set the selected indices to NaN
    flat_array[indices] = np.nan
    
    # Reshape back to original shape
    outputX = flat_array.reshape((n, p))

    # Define threshold
    thresh = 1

    try:
        nFeaturesRisk = np.sum(np.sum(np.isnan(outputX), axis=0) >= n - thresh)
    except TypeError:
        nFeaturesRisk = 0
    
    # Check for patterns with all values (or almost all) missing
    try:
        nPatternsRisk = np.sum(np.sum(np.isnan(outputX), axis=1) >= p - thresh)
    except TypeError:
        nPatternsRisk = 0
    
    # Raise warnings if necessary
    if nFeaturesRisk > 0:
        warnings.warn(f"FEATURES at risk of being all NaN: {nFeaturesRisk}")
    if nPatternsRisk > 0:
        warnings.warn(f"PATTERNS at risk of being all NaN: {nPatternsRisk}")

    return outputX

def imputation_rmse(clean_data, imputed_data, missing_mask):
    """
    Calculate Root Mean Square Error (RMSE) for imputed values
    
    Parameters:
    -----------
    clean_data : numpy.ndarray
        The original, complete dataset
    imputed_data : numpy.ndarray
        The dataset after imputation
    missing_mask : numpy.ndarray, optional
        Mask indicating missing values in the original dataset.
    
    Returns:
    --------
    float
        Root Mean Square Error for the imputed values
    dict
        Detailed performance metrics
    """
    # Validate input shapes
    if clean_data.shape != imputed_data.shape:
        raise ValueError("Clean and imputed datasets must have the same shape")
    
    # Ensure mask is boolean
    missing_mask = missing_mask.astype(bool)
    
    # Calculate errors only for missing values
    errors = clean_data[missing_mask] - imputed_data[missing_mask]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # Additional performance metrics
    metrics = {
        'rmse': rmse,
        'mae': np.mean(np.abs(errors)),
        'total_missing': np.sum(missing_mask),
        'missing_percentage': np.sum(missing_mask) / missing_mask.size * 100,
        'min_error': np.abs(np.min(errors)),
        'max_error': np.abs(np.max(errors)),
        'std_error': np.std(errors)
    }
    
    return metrics

def custom_imputation_error_onehot(clean_data, imputed_data, missing_mask, column_types, onehot_indices):
    """
    Calculate a weighted error metric for imputed values with one-hot encoding for categorical columns.
    
    Parameters:
    -----------
    clean_data : numpy.ndarray
        The original, complete dataset (in one-hot format for categorical columns).
    imputed_data : numpy.ndarray
        The dataset after imputation (in one-hot format for categorical columns).
    missing_mask : numpy.ndarray
        Mask indicating missing values in the original dataset.
    column_types : list of str
        List indicating the type of each original column: 'categorical' or 'numerical'.
    onehot_indices : dict
        A dictionary where keys are the indices of categorical columns in the original dataset,
        and values are the indices of their corresponding one-hot-encoded columns.
    
    Returns:
    --------
    dict
        Combined weighted error and individual errors.
    """
    # Validate input
    if clean_data.shape != imputed_data.shape:
        raise ValueError("Clean and imputed datasets must have the same shape")
    
    missing_mask = missing_mask.astype(bool)
    n_cols = len(column_types)
    
    # Separate weights for categorical and numerical columns
    n_cat = sum(1 for t in column_types if t == 'categorical')
    n_num = sum(1 for t in column_types if t == 'numerical')
    
    total_columns = n_cat + n_num
    w_cat = n_cat / total_columns if n_cat > 0 else 0
    w_num = n_num / total_columns if n_num > 0 else 0
    
    # Cross-entropy for categorical columns
    ce = 0
    if n_cat > 0:
        for col_idx, onehot_range in onehot_indices.items():
            # Get indices for one-hot encoded columns
            onehot_cols = onehot_range
            cat_clean = clean_data[:, onehot_cols]
            cat_imputed = imputed_data[:, onehot_cols]
            cat_missing_mask = missing_mask[:, onehot_cols[0]]  # Check first column in one-hot range
            
            # Filter rows with missing values
            if np.any(cat_missing_mask):
                clean_rows = cat_clean[cat_missing_mask]
                imputed_rows = cat_imputed[cat_missing_mask]
                ce += log_loss(clean_rows, imputed_rows)
        ce /= n_cat  # Average cross-entropy across categorical columns
    
    # RMSE for numerical columns
    rmse = 0
    if n_num > 0:
        num_indices = [i for i, t in enumerate(column_types) if t == 'numerical']
        num_clean = clean_data[:, num_indices]
        num_imputed = imputed_data[:, num_indices]
        num_missing_mask = missing_mask[:, num_indices]
        
        errors = num_clean[num_missing_mask] - num_imputed[num_missing_mask]
        rmse = np.sqrt(np.mean(errors**2))
    
    # Weighted sum of errors
    weighted_error = w_cat * ce + w_num * rmse
    
    return {
        'weighted_error': weighted_error,
        'categorical_error': ce,
        'numerical_error': rmse,
        'weights': {'categorical': w_cat, 'numerical': w_num}
    }



def plot_imputation_errors(clean_data, imputed_data, missing_mask=None):
    """
    Create visualizations of imputation errors
    """
    
    # Create missing mask if not provided
    if missing_mask is None:
        missing_mask = np.isnan(clean_data)
    
    # Calculate errors
    errors = clean_data[missing_mask] - imputed_data[missing_mask]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of errors
    ax1.hist(errors, bins=30, edgecolor='black')
    ax1.set_title('Distribution of Imputation Errors')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    
    # Scatter plot of true vs imputed values
    ax2.scatter(clean_data[missing_mask], imputed_data[missing_mask], alpha=0.5)
    ax2.plot([clean_data[missing_mask].min(), clean_data[missing_mask].max()], 
             [clean_data[missing_mask].min(), clean_data[missing_mask].max()], 
             'r--', lw=2)
    ax2.set_title('True vs Imputed Values')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Imputed Values')
    
    plt.tight_layout()
    plt.show()
  

  