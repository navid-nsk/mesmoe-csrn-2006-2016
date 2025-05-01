import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib import patches
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def visualize_pde_parameters_on_map(visualizer, interpretation_dir, shapefile_path, output_dir):
    """
    Create map-based visualizations for PDE parameters (velocity, diffusion, amplification)
    aggregating data across all batches and DAUIDs, separated by ethnicity.
    
    Args:
        visualizer: The ModelInterpretationVisualizer instance
        interpretation_dir (str): Directory containing interpretation JSON files
        shapefile_path (str): Path to the shapefile with DA boundaries
        output_dir (str): Directory to save the visualizations
        
    Returns:
        bool: Success status
    """
    try:
        # Create output directory for PDE map visualizations
        pde_map_dir = os.path.join(output_dir, "pde_map_visualizations")
        os.makedirs(pde_map_dir, exist_ok=True)
        
        # Load shapefile
        logger.info(f"Loading shapefile from {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} geometries")
        except Exception as e:
            logger.error(f"Error loading shapefile: {str(e)}")
            return False
        
        # Find DAUID column in shapefile
        dauid_column = None
        for column in gdf.columns:
            if column.upper() == 'DAUID' or 'DAUID' in column.upper():
                dauid_column = column
                break
        
        if dauid_column is None:
            logger.warning("DAUID column not found in shapefile, trying common ID field names")
            # Try common ID field names in shapefiles
            for column in ['ID', 'GEOID', 'AREA_ID', 'DA_ID', 'DISSEMINATION_AREA_ID']:
                if column in gdf.columns:
                    dauid_column = column
                    logger.info(f"Using {dauid_column} as the DA identifier")
                    break
            
            if dauid_column is None:
                logger.error("No suitable ID column found in shapefile")
                logger.info(f"Available columns in shapefile: {list(gdf.columns)}")
                return False
        
        # Ensure DAUID is string type for consistent joining
        gdf[dauid_column] = gdf[dauid_column].astype(str)
        
        # Find all interpretation JSON files
        json_files = [f for f in os.listdir(interpretation_dir) 
                    if f.endswith('.json') and 'interpretation' in f]
        
        if not json_files:
            logger.warning(f"No interpretation files found in {interpretation_dir}")
            return False
        
        logger.info(f"Found {len(json_files)} interpretation files")
        
        # Initialize data structures to store aggregated PDE parameters by DAUID and ethnicity
        pde_params = {
            'diffusion': defaultdict(lambda: defaultdict(list)),
            'amplification': defaultdict(lambda: defaultdict(list)),
            'velocity_x': defaultdict(lambda: defaultdict(list)),
            'velocity_y': defaultdict(lambda: defaultdict(list)),
            'velocity_magnitude': defaultdict(lambda: defaultdict(list))
        }
        
        # Maximum number of ethnicities encountered
        max_ethnicities = 0
        ethnicity_names = visualizer.ethnicity_names if visualizer.ethnicity_names else None
        
        # Process each interpretation file
        for json_file in json_files:
            file_path = os.path.join(interpretation_dir, json_file)
            
            # Load interpretation data
            interpretation_data = visualizer.load_interpretation_data(file_path)
            if interpretation_data is None or 'pde_expert' not in interpretation_data:
                logger.warning(f"No PDE expert data found in {json_file}, skipping")
                continue
            
            pde_data = interpretation_data['pde_expert']
            
            # Find dauid_list if available
            dauid_list = interpretation_data.get('dauid_list', [])
            
            # DEBUG: Print the structure of pde_data
            logger.info(f"PDE data structure in {json_file}:")
            logger.info(f"Keys in pde_data: {list(pde_data.keys())}")
            
            # Check if there's a diffusion key and print its type
            if 'diffusion' in pde_data:
                logger.info(f"Type of diffusion: {type(pde_data['diffusion'])}")
                if isinstance(pde_data['diffusion'], dict):
                    logger.info(f"Number of DAUIDs in diffusion: {len(pde_data['diffusion'])}")
                    # Print first DAUID key and its data
                    if pde_data['diffusion']:
                        first_dauid = next(iter(pde_data['diffusion']))
                        logger.info(f"First DAUID: {first_dauid}")
                        logger.info(f"Type of diffusion data for first DAUID: {type(pde_data['diffusion'][first_dauid])}")
                        logger.info(f"Shape or length of diffusion data for first DAUID: {len(pde_data['diffusion'][first_dauid]) if hasattr(pde_data['diffusion'][first_dauid], '__len__') else 'N/A'}")
            
            # Check for PDE parameters by DAUID
            if ('diffusion' in pde_data and isinstance(pde_data['diffusion'], dict) or
                'amplification' in pde_data and isinstance(pde_data['amplification'], dict) or
                'velocity' in pde_data and isinstance(pde_data['velocity'], dict)):
                
                # Get all DAUIDs in this interpretation
                if 'diffusion' in pde_data and isinstance(pde_data['diffusion'], dict):
                    dauid_keys = list(pde_data['diffusion'].keys())
                elif 'amplification' in pde_data and isinstance(pde_data['amplification'], dict):
                    dauid_keys = list(pde_data['amplification'].keys())
                elif 'velocity' in pde_data and isinstance(pde_data['velocity'], dict):
                    dauid_keys = list(pde_data['velocity'].keys())
                else:
                    dauid_keys = []
                
                # DEBUG: Print number of DAUIDs
                logger.info(f"Number of unique DAUIDs: {len(dauid_keys)}")
                
                # Process each DAUID
                for dauid in dauid_keys:
                    # Extract PDE parameters for this DAUID
                    diffusion = np.array(pde_data['diffusion'][dauid]) if 'diffusion' in pde_data and dauid in pde_data['diffusion'] else None
                    amplification = np.array(pde_data['amplification'][dauid]) if 'amplification' in pde_data and dauid in pde_data['amplification'] else None
                    velocity = np.array(pde_data['velocity'][dauid]) if 'velocity' in pde_data and dauid in pde_data['velocity'] else None
                    
                    # DEBUG: Print parameter values for this DAUID
                    if dauid == dauid_keys[0]:  # Only for the first DAUID to avoid excessive logs
                        logger.info(f"Parameters for DAUID {dauid}:")
                        if diffusion is not None:
                            logger.info(f"  Diffusion: {diffusion}")
                        if amplification is not None:
                            logger.info(f"  Amplification: {amplification}")
                        if velocity is not None:
                            logger.info(f"  Velocity: {velocity}")
                    
                    # Update max ethnicities
                    if diffusion is not None and len(diffusion) > max_ethnicities:
                        max_ethnicities = len(diffusion)
                    elif amplification is not None and len(amplification) > max_ethnicities:
                        max_ethnicities = len(amplification)
                    elif velocity is not None and velocity.shape[0] > max_ethnicities:
                        max_ethnicities = velocity.shape[0]
                    
                    # Store parameters by ethnicity
                    for eth_idx in range(max_ethnicities):
                        # Store diffusion
                        if diffusion is not None and eth_idx < len(diffusion):
                            pde_params['diffusion'][dauid][eth_idx].append(float(diffusion[eth_idx]))
                        
                        # Store amplification
                        if amplification is not None and eth_idx < len(amplification):
                            pde_params['amplification'][dauid][eth_idx].append(float(amplification[eth_idx]))
                        
                        # Store velocity components and magnitude
                        if velocity is not None and eth_idx < velocity.shape[0]:
                            vx = float(velocity[eth_idx, 0])
                            vy = float(velocity[eth_idx, 1])
                            magnitude = np.sqrt(vx**2 + vy**2)
                            
                            pde_params['velocity_x'][dauid][eth_idx].append(vx)
                            pde_params['velocity_y'][dauid][eth_idx].append(vy)
                            pde_params['velocity_magnitude'][dauid][eth_idx].append(magnitude)
        
        # If no ethnicity names were provided, create default ones
        if ethnicity_names is None or len(ethnicity_names) < max_ethnicities:
            ethnicity_names = [f'Ethnicity_{i}' for i in range(max_ethnicities)]
        
        # Calculate averages for each parameter by DAUID and ethnicity
        avg_params = {
            'diffusion': defaultdict(dict),
            'amplification': defaultdict(dict),
            'velocity_x': defaultdict(dict),
            'velocity_y': defaultdict(dict),
            'velocity_magnitude': defaultdict(dict)
        }
        
        # DEBUG: Print information before averaging
        for param_name in pde_params.keys():
            logger.info(f"Parameter: {param_name}")
            logger.info(f"  Number of DAUIDs with data: {len(pde_params[param_name])}")
            
            # Check if all DAUIDs have the same ethnic data
            sample_dauid = next(iter(pde_params[param_name])) if pde_params[param_name] else None
            if sample_dauid:
                sample_eth_data = pde_params[param_name][sample_dauid]
                eth_indices = list(sample_eth_data.keys())
                logger.info(f"  Ethnicity indices present: {eth_indices}")
                
                # Check a few DAUIDs to see if they have the same values
                dauid_sample = list(pde_params[param_name].keys())[:min(5, len(pde_params[param_name]))]
                logger.info(f"  Checking sample of DAUIDs: {dauid_sample}")
                
                for eth_idx in eth_indices:
                    logger.info(f"  Values for ethnicity {eth_idx}:")
                    for dauid in dauid_sample:
                        if eth_idx in pde_params[param_name][dauid]:
                            values = pde_params[param_name][dauid][eth_idx]
                            logger.info(f"    DAUID {dauid}: {values}")
        
        for param_name in avg_params.keys():
            for dauid in pde_params[param_name]:
                for eth_idx in pde_params[param_name][dauid]:
                    if pde_params[param_name][dauid][eth_idx]:  # Only calculate if there's data
                        avg_params[param_name][dauid][eth_idx] = np.mean(pde_params[param_name][dauid][eth_idx])
        
        # DEBUG: Print some of the calculated averages
        for param_name in avg_params.keys():
            logger.info(f"Averages for {param_name}:")
            sample_dauids = list(avg_params[param_name].keys())[:min(5, len(avg_params[param_name]))]
            for dauid in sample_dauids:
                logger.info(f"  DAUID {dauid}: {avg_params[param_name][dauid]}")
                
            # Check if all DAUIDs have the same value for each ethnicity
            if sample_dauids:
                for eth_idx in range(max_ethnicities):
                    values_by_dauid = [avg_params[param_name][dauid].get(eth_idx, None) for dauid in avg_params[param_name]]
                    values_by_dauid = [v for v in values_by_dauid if v is not None]
                    
                    if values_by_dauid:
                        all_same = all(abs(v - values_by_dauid[0]) < 1e-6 for v in values_by_dauid)
                        logger.info(f"  All DAUIDs have same value for ethnicity {eth_idx}: {all_same}")
                        logger.info(f"  Number of unique values: {len(set(round(v, 6) for v in values_by_dauid))}")
                        logger.info(f"  Min value: {min(values_by_dauid)}, Max value: {max(values_by_dauid)}")
        
        # Create visualizations for each ethnicity and parameter
        for eth_idx in range(max_ethnicities):
            eth_name = ethnicity_names[eth_idx] if eth_idx < len(ethnicity_names) else f'Ethnicity_{eth_idx}'
        
        # Create visualizations for each ethnicity and parameter
        for eth_idx in range(max_ethnicities):
            eth_name = ethnicity_names[eth_idx] if eth_idx < len(ethnicity_names) else f'Ethnicity_{eth_idx}'
            
            # Create visualizations for each parameter
            for param_name in ['diffusion', 'amplification', 'velocity_magnitude']:
                # Create a copy of the GeoDataFrame for this visualization
                plot_gdf = gdf.copy()
                
                # Add parameter values to GeoDataFrame
                plot_gdf[param_name] = np.nan
                
                # Fill in values from our aggregated data
                for dauid in avg_params[param_name]:
                    if eth_idx in avg_params[param_name][dauid]:
                        value = avg_params[param_name][dauid][eth_idx]
                        
                        # Find matching rows in GeoDataFrame
                        matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                        if not matching_rows.empty:
                            plot_gdf.loc[matching_rows.index, param_name] = value
                
                # Create the visualization
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Choose colormap based on parameter
                if param_name == 'diffusion':
                    cmap = 'Blues'
                    title = f'Average Diffusion Coefficient for {eth_name}'
                elif param_name == 'amplification':
                    cmap = 'Oranges'
                    title = f'Average Amplification Factor for {eth_name}'
                else:  # velocity_magnitude
                    cmap = 'Reds'
                    title = f'Average Velocity Magnitude for {eth_name}'
                
                # Plot the parameter values
                plot_gdf.plot(
                    column=param_name,
                    ax=ax,
                    legend=True,
                    cmap=cmap,
                    missing_kwds={'color': 'lightgrey'},
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.8
                )
                
                # Add title and styling
                ax.set_title(title, fontsize=16)
                ax.set_axis_off()
                
                # Add a note about missing data
                missing_count = plot_gdf[param_name].isna().sum()
                if missing_count > 0:
                    ax.text(
                        0.05, 0.05, 
                        f"Missing data: {missing_count}/{len(plot_gdf)} areas", 
                        transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                
                # Save the figure
                plt.tight_layout()
                save_path = os.path.join(pde_map_dir, f'pde_{param_name}_{eth_name.replace(" ", "_")}.png')
                plt.savefig(save_path)
                plt.close()
                
                logger.info(f"Created {param_name} map for {eth_name}")
            
            # Create velocity vector field visualization
            # This is more complex as we need both x and y components
            plot_gdf = gdf.copy()
            
            # Add centroid coordinates for plotting vectors
            plot_gdf['centroid_x'] = plot_gdf.geometry.centroid.x
            plot_gdf['centroid_y'] = plot_gdf.geometry.centroid.y
            
            # Add velocity components
            plot_gdf['velocity_x'] = np.nan
            plot_gdf['velocity_y'] = np.nan
            
            # Fill in values from our aggregated data
            for dauid in avg_params['velocity_x']:
                if eth_idx in avg_params['velocity_x'][dauid] and eth_idx in avg_params['velocity_y'][dauid]:
                    vx = avg_params['velocity_x'][dauid][eth_idx]
                    vy = avg_params['velocity_y'][dauid][eth_idx]
                    
                    # Find matching rows in GeoDataFrame
                    matching_rows = plot_gdf[plot_gdf[dauid_column] == dauid]
                    if not matching_rows.empty:
                        plot_gdf.loc[matching_rows.index, 'velocity_x'] = vx
                        plot_gdf.loc[matching_rows.index, 'velocity_y'] = vy
            
            # Drop rows with missing velocity data
            vector_gdf = plot_gdf.dropna(subset=['velocity_x', 'velocity_y'])
            
            if not vector_gdf.empty:
                # Create the visualization
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Plot the base map
                plot_gdf.plot(
                    ax=ax,
                    color='lightgrey',
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.5
                )
                
                # Calculate velocities for scaling the vectors
                velocities = np.sqrt(vector_gdf['velocity_x']**2 + vector_gdf['velocity_y']**2)
                max_velocity = velocities.max()
                min_velocity = velocities.min()
                
                # Normalize for color mapping
                norm = mcolors.Normalize(vmin=min_velocity, vmax=max_velocity)
                
                # Plot velocity vectors at each DA centroid
                q = ax.quiver(
                    vector_gdf['centroid_x'], 
                    vector_gdf['centroid_y'],
                    vector_gdf['velocity_x'], 
                    vector_gdf['velocity_y'],
                    velocities,
                    cmap='viridis',
                    norm=norm,
                    scale=20,  # Adjust scale as needed
                    width=0.002,  # Adjust width as needed
                    headwidth=5,
                    headlength=7
                )
                
                # Add colorbar
                cbar = plt.colorbar(q, ax=ax)
                cbar.set_label('Velocity Magnitude')
                
                # Add title and styling
                ax.set_title(f'Velocity Vector Field for {eth_name}', fontsize=16)
                ax.set_axis_off()
                
                # Add a note about data coverage
                missing_count = plot_gdf.shape[0] - vector_gdf.shape[0]
                if missing_count > 0:
                    ax.text(
                        0.05, 0.05, 
                        f"Areas with velocity data: {vector_gdf.shape[0]}/{plot_gdf.shape[0]}", 
                        transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                
                # Save the figure
                plt.tight_layout()
                save_path = os.path.join(pde_map_dir, f'pde_velocity_vector_{eth_name.replace(" ", "_")}.png')
                plt.savefig(save_path)
                plt.close()
                
                logger.info(f"Created velocity vector field map for {eth_name}")
                
                # Create a simplified flow direction map
                # This shows the dominant flow direction as colors
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Plot the base map
                plot_gdf.plot(
                    ax=ax,
                    color='lightgrey',
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.5
                )
                
                # Calculate direction angles in degrees (0 = east, 90 = north)
                vector_gdf['direction'] = np.degrees(np.arctan2(vector_gdf['velocity_y'], vector_gdf['velocity_x']))
                
                # Create a custom colormap for directions
                direction_cmap = cm.hsv
                norm = mcolors.Normalize(vmin=-180, vmax=180)
                
                # Plot direction as colored points
                scatter = ax.scatter(
                    vector_gdf['centroid_x'], 
                    vector_gdf['centroid_y'],
                    c=vector_gdf['direction'],
                    cmap=direction_cmap,
                    norm=norm,
                    s=velocities*100,  # Size by velocity magnitude
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Flow Direction (degrees)')
                
                # Add direction legend
                legend_elements = [
                    patches.Patch(facecolor=direction_cmap(norm(0)), label='East'),
                    patches.Patch(facecolor=direction_cmap(norm(90)), label='North'),
                    patches.Patch(facecolor=direction_cmap(norm(180)), label='West'),
                    patches.Patch(facecolor=direction_cmap(norm(-90)), label='South')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                # Add title and styling
                ax.set_title(f'Flow Direction for {eth_name}', fontsize=16)
                ax.set_axis_off()
                
                # Save the figure
                plt.tight_layout()
                save_path = os.path.join(pde_map_dir, f'pde_flow_direction_{eth_name.replace(" ", "_")}.png')
                plt.savefig(save_path)
                plt.close()
                
                logger.info(f"Created flow direction map for {eth_name}")
        
        logger.info(f"Successfully created PDE parameter map visualizations")
        return True
        
    except Exception as e:
        logger.error(f"Error creating PDE parameter map visualizations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False