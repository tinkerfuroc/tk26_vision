import json
import numpy as np
from pan_tilt_model import forward_kinematics_fixed
from utils import pose_to_matrix
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def pose_error(T_pred, T_gt):
    """Compute error between two transforms."""
    # Translation error
    trans_error = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])

    # Rotation error (angle difference)
    R_pred = T_pred[:3, :3]
    R_gt = T_gt[:3, :3]
    R_err = R_gt.T @ R_pred
    angle_error = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

    return trans_error + 0.1 * angle_error  # Can tune weight balance if needed

def load_measurements(path):
    with open(path, 'r') as f:
        return json.load(f)

def objective(params, data, L_pan, L_tilt):
    total_error = 0.0
    # params is 9D: [p0(3), r2(3), p2(3)]
    # Construct the full 12D params for forward_kinematics_fixed, assuming r0 is zero
    full_params = np.concatenate((np.zeros(3), params))
    for sample in data:
        theta_pan = np.deg2rad(sample["theta_pan"])
        theta_tilt = np.deg2rad(sample["theta_tilt"])
        T_gt = pose_to_matrix(sample["translation"], sample["rotation"])
        T_pred = forward_kinematics_fixed(theta_pan, theta_tilt, full_params, L_pan, L_tilt)
        total_error += pose_error(T_pred, T_gt)
    return total_error

def print_results(res):
    """Prints optimization results in a readable format."""
    print("\n--- Optimization Results (Simplified Model) ---")
    print(f"Success: {res.success}")
    if not res.success:
        print(f"Message: {res.message}")
    print(f"Final Objective Value (Total Error): {res.fun:.4f}")

    opt_params = res.x
    # r0 is zero by definition in this simplified model
    r0 = np.zeros(3)
    p0 = opt_params[0:3]
    r2 = opt_params[3:6]
    p2 = opt_params[6:9]

    rot0 = Rotation.from_rotvec(r0)
    rot2 = Rotation.from_rotvec(r2)

    rot0_euler_deg = rot0.as_euler('xyz', degrees=True)
    rot2_euler_deg = rot2.as_euler('xyz', degrees=True)

    rot0_quat = rot0.as_quat()
    rot2_quat = rot2.as_quat()

    print("\n--- Calibrated Transformations ---")
    print("1. base_link -> pan_link (T0):")
    print(f"  - Translation (x,y,z) [m]: {np.round(p0, 4)}")
    print(f"  - Rotation (Euler xyz) [deg]: {np.round(rot0_euler_deg, 4)} (fixed)")
    print(f"  - Rotation (Quaternion xyzw): {np.round(rot0_quat, 4)} (fixed)")

    print("\n2. tilt_link -> camera_link (T3):")
    print(f"  - Translation (x,y,z) [m]: {np.round(p2, 4)}")
    print(f"  - Rotation (Euler xyz) [deg]: {np.round(rot2_euler_deg, 4)}")
    print(f"  - Rotation (Quaternion xyzw): {np.round(rot2_quat, 4)}")
    print("------------------------------------")

def main():
    L_pan = 0.056  # in meters, from blueprint (56mm)
    L_tilt = 0.06582  # in meters, from blueprint (65.82mm)
    print("Using fixed pan/tilt lengths: L_pan = {}, L_tilt = {}".format(L_pan, L_tilt))
    data = load_measurements("data/measurements.json")
    print("Loaded {} samples".format(len(data)))
    
    # Parameters are now 9D: [p0(3), r2(3), p2(3)]
    init_params = np.zeros(9)
    # Initial guess for p0 (base_link -> pan_link translation)
    init_params[0:3] = [-0.2, -0.2, 1.3]
    
    print("Initial params (9D):", init_params)
    
    # Add bounds to constrain the optimization
    bounds = [(-np.inf, np.inf)] * 9
    # Set bounds for the z-component of p2 (tilt_link -> camera_link translation)
    # Parameters are [p0(3), r2(3), p2(3)]
    bounds[8] = (-0.05, 0.05)
    
    res = minimize(objective, init_params, args=(data, L_pan, L_tilt), method='L-BFGS-B', bounds=bounds)
    print_results(res)
    
    # Save the full 12 parameters for compatibility, with r0 as zero
    full_opt_params = np.concatenate((np.zeros(3), res.x))
    np.save("opt_params_simple.npy", full_opt_params)

if __name__ == "__main__":
    main()
    print("Optimization complete. Parameters saved to opt_params_simple.npy.") 