# Autodesk Fusion 360 Geometry Import Script
# Generates assembly components natively from mtcp_sensors.json.

import adsk.core, adsk.fusion, traceback
import json
import os

# Fusion 360 runs this script outside the Python package, so it can't resolve
# relative paths. Set GLOVE_REPO_ROOT in your environment or edit below.
REPO_ROOT = os.environ.get("GLOVE_REPO_ROOT", "")
JSON_PATH = os.path.join(REPO_ROOT, "software/configs/calibration/theoretical_mtcp.json")

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        
        if not design:
            ui.messageBox('No active Fusion design', 'No Design')
            return
            
        if not os.path.exists(JSON_PATH):
            ui.messageBox(f'Could not find {JSON_PATH}. Please run generate_theoretical_mtcp.py first.', 'File Missing')
            return
            
        with open(JSON_PATH, 'r') as f:
            config = json.load(f)
            
        root = design.rootComponent
        
        updated_count = 0
        created_count = 0
        
        for name, mat in config.items():
            if not name.startswith("T_mtcp_"):
                continue
            
            sensor_name = name.replace("T_mtcp_", "")
            comp_name = f"MTCP_{sensor_name.upper()}_ORIGIN"
            
            # 1. Create a flattened matrix scaled to Fusion's centimeters
            transform = adsk.core.Matrix3D.create()
            flat_array = []
            for r in range(4):
                for c in range(4):
                    val = mat[r][c]
                    if c == 3 and r < 3: 
                        val = val * 100.0 
                    flat_array.append(val)
                    
            transform.setWithArray(flat_array)
            
            # 2. Check if component exists
            existing_occ = None
            for occ in root.occurrences:
                if occ.component.name == comp_name:
                    existing_occ = occ
                    break
                    
            if existing_occ:
                # If you manually grounded it, the script can't move it!
                if existing_occ.isGrounded:
                    ui.messageBox(f'Could not update {comp_name} because you Grounded it! Please unground it first, or ignore this if intentional.', 'Component Locked')
                    continue
                # Update existing
                existing_occ.transform = transform
                updated_count += 1
            else:
                # Create new
                new_occ = root.occurrences.addNewComponent(transform)
                new_occ.component.name = comp_name
                created_count += 1
            
        ui.messageBox(f'Glove V3 Design Context Updated!\n\nCreated: {created_count}\nUpdated: {updated_count}\n\nLive from: theoretical_mtcp.json', 'MTCP Synced')

    except Exception as e:
        if ui:
            ui.messageBox('Failed to generate MTCP assembly:\n{}'.format(traceback.format_exc()))
