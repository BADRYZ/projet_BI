import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import tensorflow as tf


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

def missing_method(raw_data, mechanism='mcar', method='uniform', missing_threshold=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    data = raw_data.copy()
    rows, cols = data.shape
    t = missing_threshold

    if mechanism == 'mcar':
        v = np.random.uniform(size=(rows, cols))
        if method == 'uniform':
            mask = v <= t
        elif method == 'random':
            c = np.zeros(cols, dtype=bool)
            c[np.random.choice(cols, cols // 2, replace=False)] = True
            mask = (v <= t) & c[np.newaxis, :]
        else:
            raise ValueError(f"Unknown method: {method}")
    elif mechanism == 'mnar':
        sample_cols = np.random.choice(cols, 2, replace=False)
        m1, m2 = np.median(data[:, sample_cols], axis=0)
        v = np.random.uniform(size=(rows, cols))
        m = (data[:, sample_cols[0]] <= m1) & (data[:, sample_cols[1]] >= m2)
        mask = v <= t
        if method == 'uniform':
            mask &= m[:, np.newaxis]
        elif method == 'random':
            c = np.zeros(cols, dtype=bool)
            c[np.random.choice(cols, cols // 2, replace=False)] = True
            mask &= m[:, np.newaxis] & c[np.newaxis, :]
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    data[mask] = np.nan
    return data, mask

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

def imputation_ce_rmse(clean_data, imputed_data, missing_mask, column_types):
    """
    Calculate a weighted error metric for imputed values, combining cross-entropy for categorical columns
    and RMSE for numerical columns.
    
    Parameters:
    -----------
    clean_data : numpy.ndarray
        The original, complete dataset.
    imputed_data : numpy.ndarray
        The dataset after imputation.
    missing_mask : numpy.ndarray
        Mask indicating missing values in the original dataset.
    column_types : list of str
        List indicating the type of each column: 'categorical' or 'numerical'.
    
    Returns:
    --------
    dict
        Combined weighted error and individual errors.
    """
    # Validate input
    if clean_data.shape != imputed_data.shape:
        raise ValueError("Clean and imputed datasets must have the same shape")
    
    missing_mask = missing_mask.astype(bool)
    n_cols = clean_data.shape[1]
    
    # Separate categorical and numerical columns
    cat_indices = [i for i, t in enumerate(column_types) if t == 'categorical']
    num_indices = [i for i, t in enumerate(column_types) if t == 'numerical']
    
    n_cat = len(cat_indices)
    n_num = len(num_indices)
    
    # Weights
    total_columns = n_cat + n_num
    w_cat = n_cat / total_columns if n_cat > 0 else 0
    w_num = n_num / total_columns if n_num > 0 else 0
    
    # Cross-entropy for categorical columns
    ce = 0
    if n_cat > 0:
        cat_clean = clean_data[:, cat_indices]
        cat_imputed = imputed_data[:, cat_indices]
        cat_missing_mask = missing_mask[:, cat_indices]
        
        for i in range(n_cat):
            clean = cat_clean[cat_missing_mask[:, i], i]
            imputed = cat_imputed[cat_missing_mask[:, i], i]
            if clean.size > 0:
                # Use log_loss for cross-entropy (requires one-hot or probability input for imputed data)
                ce += log_loss(clean, imputed, labels=np.unique(clean))
        ce /= n_cat  # Average cross-entropy across categorical columns
    
    # RMSE for numerical columns
    rmse = 0
    if n_num > 0:
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
  

  