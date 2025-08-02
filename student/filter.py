# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.size = int(params.dim_state/2)      # 3D objects

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        F = np.identity(params.dim_state) #6x6
        F[0,3] = params.dt
        F[1,4] = params.dt
        F[2,5] = params.dt
        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        I = np.identity(self.size) #3x3
        Q3 = (params.dt**3 * params.q / 3) * I
        Q2 = (params.dt**2 * params.q / 2) * I
        Q1 = (params.dt * params.q) * I
        return np.vstack((np.hstack((Q3, Q2)),
                          np.hstack((Q2, Q1))))
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        x = F @ track.x
        P = F @ track.P @ F.T + self.Q()

        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        H = meas.sensor.get_H(track.x)
        K = track.P @ H.T @ np.linalg.inv(self.S(track, meas, H))
        x = track.x + K @ self.gamma(track, meas)
        P = (np.identity(params.dim_state) - K @ H) @ track.P

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        return meas.z - meas.sensor.get_hx(track.x)
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return H @ track.P @ H.T + meas.R
        
        ############
        # END student code
        ############ 