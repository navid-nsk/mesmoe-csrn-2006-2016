import os
import logging
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse

from csrn.data_loader.spatial_data_loader import MultiEthnicDataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ethnicity_interaction_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_interpretation_data(interpretation_dir):
    """
    Load interpretation data from JSON files in the specified directory
    
    Args:
        interpretation_dir (str): Directory containing interpretation JSON files
        
    Returns:
        list: List of interpretation data dictionaries
    """
    interpretations = []
    
    # Find all interpretation JSON files
    json_files = [f for f in os.listdir(interpretation_dir) 
                 if f.endswith('.json') and 'interpretation' in f]
    
    if not json_files:
        logger.warning(f"No interpretation files found in {interpretation_dir}")
        return interpretations
    
    logger.info(f"Found {len(json_files)} interpretation files")
    
    # Load each file
    for json_file in json_files:
        file_path = os.path.join(interpretation_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            interpretations.append(data)
            logger.info(f"Loaded interpretation data from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    return interpretations

def load_da_coordinates(da_coordinates_path):
    """
    Load DA coordinates from CSV file
    
    Args:
        da_coordinates_path (str): Path to the CSV file with DA coordinates
        
    Returns:
        dict: Mapping from DAUID to (x,y) coordinates
    """
    if not os.path.exists(da_coordinates_path):
        logger.warning(f"DA coordinates file not found: {da_coordinates_path}")
        return {}
    
    try:
        da_coords_df = pd.read_csv(da_coordinates_path)
        logger.info(f"Loaded DA coordinates from {da_coordinates_path} with {len(da_coords_df)} rows")
        
        # Create mapping from DAUID to (X, Y) coordinates
        dauid_to_coords = {}
        for _, row in da_coords_df.iterrows():
            if 'DAUID' in row and 'X_axis' in row and 'Y_axis' in row:
                dauid_to_coords[str(int(float(row['DAUID'])))] = (row['X_axis'], row['Y_axis'])
            elif 'DAUID' in row and 'X' in row and 'Y' in row:
                dauid_to_coords[str(row['DAUID'])] = (row['X'], row['Y'])
        
        logger.info(f"Created coordinate mapping for {len(dauid_to_coords)} DAUIDs")
        return dauid_to_coords
    except Exception as e:
        logger.error(f"Error loading DA coordinates: {str(e)}")
        return {}

def prepare_interaction_data(interpretations):
    """
    Process and prepare ethnicity interaction data from multiple interpretations
    
    Args:
        interpretations (list): List of interpretation data dictionaries
        
    Returns:
        dict: Processed interaction data
    """
    # Initialize processed data
    processed_data = {
        'interaction_weights': None,
        'interaction_effect': None,
        'ethnicity_interactions_by_dauid': {},
        'interaction_effect_by_dauid': {},
        'dauid_list': []
    }
    
    # Extract ethnicity names from the first interpretation if available
    ethnicity_names = None
    num_ethnicities = None
    for interp in interpretations:
        if 'model_architecture' in interp and 'num_ethnicities' in interp['model_architecture']:
            num_ethnicities = interp['model_architecture']['num_ethnicities']
            logger.info(f"Found {num_ethnicities} ethnicities in model architecture")
            break
    
    data_handler, train_loader, val_loader, test_loader, ethnicity_names = load_data(args)
    
    ethnicity_names = ethnicity_names
    # Process each interpretation
    for interp in interpretations:
        # Get interaction weights and effects
        interaction_weights = interp.get('interaction_weights')
        interaction_effect = interp.get('interaction_effect')
        
        # Get DAUID list
        if 'dauid_list' in interp:
            processed_data['dauid_list'].extend(interp['dauid_list'])
        
        # Get by-DAUID interaction data
        if 'ethnicity_interactions_by_dauid' in interp:
            for dauid, interactions in interp['ethnicity_interactions_by_dauid'].items():
                processed_data['ethnicity_interactions_by_dauid'][dauid] = interactions
        
        # Get by-DAUID effect data
        if 'interaction_effect_by_dauid' in interp:
            for dauid, effects in interp['interaction_effect_by_dauid'].items():
                processed_data['interaction_effect_by_dauid'][dauid] = effects
        
        # Store valid interaction data (just use the first valid one we find)
        if processed_data['interaction_weights'] is None and interaction_weights is not None:
            # Check if it's a batch
            if isinstance(interaction_weights, list) and len(interaction_weights) > 0:
                # Create a properly sized interaction matrix
                if num_ethnicities:
                    # Create a proper average interaction matrix for all ethnicities
                    batches = []
                    for batch in interaction_weights:
                        if isinstance(batch, list) and len(batch) == num_ethnicities:
                            # Valid batch - convert to numpy array
                            batch_array = np.array(batch)
                            if batch_array.shape == (num_ethnicities, num_ethnicities):
                                batches.append(batch_array)
                    
                    if batches:
                        # Average across valid batches
                        processed_data['interaction_weights'] = np.stack(batches)
                        logger.info(f"Processed interaction weights with shape {processed_data['interaction_weights'].shape}")
                else:
                    # Just use the first batch
                    processed_data['interaction_weights'] = np.array([np.array(interaction_weights[0])])
                    logger.info(f"Using first batch of interaction weights")
            else:
                # Single batch or direct matrix
                processed_data['interaction_weights'] = np.array([np.array(interaction_weights)])
                logger.info(f"Using direct interaction weights")
        
        # Store valid effect data
        if processed_data['interaction_effect'] is None and interaction_effect is not None:
            if isinstance(interaction_effect, list) and len(interaction_effect) > 0:
                processed_data['interaction_effect'] = np.array(interaction_effect)
            else:
                processed_data['interaction_effect'] = np.array([interaction_effect])
    
    # Deduplicate DAUID list
    processed_data['dauid_list'] = list(set(processed_data['dauid_list']))
    
    logger.info(f"Processed interaction data from {len(interpretations)} interpretations")
    logger.info(f"Found data for {len(processed_data['ethnicity_interactions_by_dauid'])} DAUIDs")
    
    return processed_data, ethnicity_names

def load_data(args):
    """
    Load the dataset using the MultiEthnicDataHandler.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: Data handler and data loaders
    """
    logger.info(f"Loading data from {args.data_dir}")
    
    # Load DA coordinates
    try:
        da_coordinates = pd.read_csv(args.da_coordinates_path)
        logger.info(f"Loaded DA coordinates from {args.da_coordinates_path}")
    except Exception as e:
        logger.warning(f"Error loading DA coordinates: {str(e)}")
        da_coordinates = None
    
    # Create data handler
    data_handler = MultiEthnicDataHandler(args.data_dir, args.spatial_dir, scaler_type=args.scaler)
    
    # Pass DA coordinates to the data handler if loaded successfully
    if da_coordinates is not None:
        data_handler.da_coordinates = da_coordinates
    
    # Load data
    data = data_handler.load_data()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_handler.create_dataloaders(
        data, batch_size=args.batch_size
    )
    
    # Get ethnicity names
    ethnicity_names = []
    if hasattr(train_loader.dataset, 'ethnicity_columns'):
        ethnicity_names = train_loader.dataset.ethnicity_columns
    else:
        # Try to get them from the columns in the population data
        if 'train_pop_2006' in data:
            ethnicity_names = [col for col in data['train_pop_2006'].columns if col != 'DAUID']
    
    logger.info(f"Loaded data with {len(ethnicity_names)} ethnicities: {ethnicity_names}")
    
    return data_handler, train_loader, val_loader, test_loader, ethnicity_names

def main(args):
    """
    Main function to generate ethnicity interaction visualizations
    
    Args:
        args: Command line arguments
    """
    logger.info("=== Starting Ethnicity Interaction Analysis ===")
    logger.info(f"Arguments: {args}")
    
    # Import the EthnicityInteractionAnalyzer after checking for the file
    try:
        from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
        logger.info("Imported EthnicityInteractionAnalyzer from ethnicity_interaction_visualization_fix")
    except ImportError:
        try:
            from ethnicity_interaction_visualization import EthnicityInteractionAnalyzer
            logger.info("Imported EthnicityInteractionAnalyzer from ethnicity_interaction_visualization")
        except ImportError:
            logger.error("Could not import EthnicityInteractionAnalyzer. Make sure the file exists in the current directory.")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load interpretation data
    interpretation_dir = args.interpretation_dir
    interpretations = load_interpretation_data(interpretation_dir)
    
    if not interpretations:
        logger.error("No interpretation data found. Exiting.")
        return
    
    # Load DA coordinates if provided
    dauid_to_coords = {}
    if args.da_coordinates_path:
        dauid_to_coords = load_da_coordinates(args.da_coordinates_path)
    
    # Process and prepare interaction data
    processed_data, default_ethnicity_names = prepare_interaction_data(interpretations)
    
    # Use provided ethnicity names if available, otherwise use defaults
    ethnicity_names = args.ethnicity_names if args.ethnicity_names else default_ethnicity_names
    
    if ethnicity_names:
        logger.info(f"Using ethnicity names: {ethnicity_names}")
    else:
        # If still no ethnicity names, create default ones
        num_ethnicities = 5  # Default if we can't determine
        
        # Try to infer number of ethnicities from interaction data
        if processed_data['interaction_weights'] is not None:
            weights_shape = processed_data['interaction_weights'].shape
            if len(weights_shape) >= 3:
                num_ethnicities = weights_shape[1]  # [batch, ethnicity, ethnicity]
            elif len(weights_shape) == 2:
                num_ethnicities = weights_shape[0]  # [ethnicity, ethnicity]
        
        ethnicity_names = [f"Ethnicity {i}" for i in range(num_ethnicities)]
        logger.info(f"Created default ethnicity names for {num_ethnicities} ethnicities")
    
    # Create custom color palette for ethnicities
    colors = plt.cm.tab10.colors
    if args.colormap:
        try:
            colors = plt.cm.get_cmap(args.colormap).colors
            logger.info(f"Using custom colormap: {args.colormap}")
        except:
            logger.warning(f"Invalid colormap: {args.colormap}, using default")
    
    # Initialize the ethnicity interaction analyzer
    analyzer = EthnicityInteractionAnalyzer(
        ethnicity_names=ethnicity_names,
        output_dir=args.output_dir,
        color_palette=colors,
        significant_threshold=args.threshold
    )
    
    # Set DAUID coordinates
    if dauid_to_coords:
        analyzer.set_dauid_coordinates(dauid_to_coords)
    
    # Extract interaction data from prepared data
    logger.info("Extracting interaction data")
    success = analyzer.extract_interaction_data(processed_data)
    
    if not success:
        logger.error("Failed to extract interaction data. Exiting.")
        return
    
    # Load shapefile if provided
    gdf = None
    if args.shapefile_path and os.path.exists(args.shapefile_path):
        try:
            gdf = gpd.read_file(args.shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} geometries")
        except Exception as e:
            logger.error(f"Error loading shapefile: {str(e)}")
    
    # Generate all visualizations
    if args.all:
        logger.info("Generating all visualizations")
        analyzer.run_all_visualizations(gdf, args.shapefile_path)
    else:
        # Generate individual visualizations based on flags
        if args.network:
            logger.info("Generating network visualizations")
            analyzer.visualize_interaction_network()
        
        if args.flow:
            logger.info("Generating flow diagram visualizations")
            analyzer.visualize_flow_diagrams()
        
        if args.spatial and (dauid_to_coords or gdf is not None):
            logger.info("Generating spatial visualizations")
            analyzer.visualize_spatial_interactions(gdf, args.shapefile_path)
        
        if args.temporal:
            logger.info("Generating temporal cascade visualizations")
            analyzer.visualize_temporal_cascade(time_steps=args.time_steps)
        
        if args.metrics:
            logger.info("Generating interaction metrics visualizations")
            analyzer.visualize_interaction_metrics()
    
    logger.info("=== Ethnicity Interaction Analysis Completed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ethnicity Interaction Analysis")
    
    # Required parameters
    parser.add_argument("--interpretation_dir", type=str, required=True,
                        help="Directory containing interpretation JSON files")
    parser.add_argument("--output_dir", type=str, default="./ethnicity_visualizations",
                        help="Directory to save ethnicity interaction visualizations")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--spatial_dir", type=str, default="./spatial_data",
                        help="Directory containing spatial data files")
    
    # Optional parameters
    parser.add_argument("--ethnicity_names", type=str, nargs='+',
                        help="List of ethnicity names")
    parser.add_argument("--da_coordinates_path", type=str, default='./spatial_data/toronto_da.csv',
                        help="Path to the CSV file containing DA coordinates")
    parser.add_argument("--shapefile_path", type=str, default='./spatial_data/da_boundary.shp',
                        help="Path to shapefile with DA boundaries for spatial visualization")
    parser.add_argument("--colormap", type=str, default=None,
                        help="Matplotlib colormap name for ethnicity colors")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for significant interactions")
    parser.add_argument("--time_steps", type=int, default=5,
                        help="Number of time steps for temporal cascade visualization")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"],
                        help="Type of scaler to use for feature normalization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    # Visualization type flags
    parser.add_argument("--all", action="store_true",
                        help="Generate all visualization types")
    parser.add_argument("--network", action="store_true",
                        help="Generate network visualizations")
    parser.add_argument("--flow", action="store_true",
                        help="Generate flow diagram visualizations")
    parser.add_argument("--spatial", action="store_true",
                        help="Generate spatial visualizations")
    parser.add_argument("--temporal", action="store_true",
                        help="Generate temporal cascade visualizations")
    parser.add_argument("--metrics", action="store_true",
                        help="Generate interaction metrics visualizations")
    
    args = parser.parse_args()
    
    # If no specific visualization is requested, enable all
    if not any([args.all, args.network, args.flow, args.spatial, args.temporal, args.metrics]):
        args.all = True
    
    main(args)