/**
 * DataCollector - Centralized data collection system
 * Stores all user interaction data in localStorage and exports to JSON
 */

const DataCollector = {
    userId: null,
    sessionData: null,
    
    /**
     * Initialize data collection for a new user
     */
    initializeUser(userId) {
        this.userId = userId;
        
        // Initialize session data structure
        const sessionData = {
            userId: userId,
            sessionStartTime: Date.now(),
            sessionStartDate: new Date().toISOString(),
            browser: navigator.userAgent,
            screenSize: {
                width: window.screen.width,
                height: window.screen.height
            },
            demographics: {},
            surveyResponses: {
                preSurvey: {},
                postSurvey: {}
            },
            simulatorData: {
                trials: [],
                totalTime: 0,
                totalMoves: 0,
                totalDistance: 0
            },
            condition: 1,
            events: [],
            pageVisits: []
        };
        
        // Store in localStorage
        localStorage.setItem(`study_data_${userId}`, JSON.stringify(sessionData));
        this.sessionData = sessionData;
        
        console.log(`Data collection initialized for user: ${userId}`);
        return sessionData;
    },
    
    /**
     * Load existing session data
     */
    loadSessionData() {
        const userId = localStorage.getItem('studyUserId');
        if (!userId) {
            console.error('No user ID found');
            return null;
        }
        
        this.userId = userId;
        const dataKey = `study_data_${userId}`;
        const dataStr = localStorage.getItem(dataKey);
        
        if (dataStr) {
            this.sessionData = JSON.parse(dataStr);
            return this.sessionData;
        } else {
            // Initialize if not found
            return this.initializeUser(userId);
        }
    },
    
    /**
     * Save current session data to localStorage
     */
    saveSessionData() {
        if (!this.sessionData || !this.userId) {
            console.error('No session data to save');
            return;
        }
        
        const dataKey = `study_data_${this.userId}`;
        localStorage.setItem(dataKey, JSON.stringify(this.sessionData));
    },
    
    /**
     * Record demographic information
     */
    recordDemographics(demographics) {
        if (!this.sessionData) this.loadSessionData();
        
        this.sessionData.demographics = {
            ...this.sessionData.demographics,
            ...demographics,
            timestamp: Date.now()
        };
        
        this.saveSessionData();
        console.log('Demographics recorded:', demographics);
    },
    
    /**
     * Record survey responses
     */
    recordSurvey(surveyType, responses) {
        if (!this.sessionData) this.loadSessionData();
        
        if (surveyType === 'pre') {
            this.sessionData.surveyResponses.preSurvey = {
                ...responses,
                timestamp: Date.now()
            };
        } else if (surveyType === 'post') {
            this.sessionData.surveyResponses.postSurvey = {
                ...responses,
                timestamp: Date.now()
            };
        }
        
        this.saveSessionData();
        console.log(`${surveyType} survey recorded:`, responses);
    },
    
    /**
     * Start a new simulator trial
     */
    startTrial(trialConfig) {
        if (!this.sessionData) this.loadSessionData();
        
        const trial = {
            trialId: this.sessionData.simulatorData.trials.length,
            startTime: Date.now(),
            config: trialConfig,
            movements: [],
            actions: [],
            probabilities: [],
            completed: false,
            completionTime: null
        };
        
        this.sessionData.simulatorData.trials.push(trial);
        this.saveSessionData();
        
        return trial.trialId;
    },
    
    /**
     * Record a movement in the current trial
     */
    recordMovement(trialId, movementData) {
        if (!this.sessionData) this.loadSessionData();
        
        const trial = this.sessionData.simulatorData.trials[trialId];
        if (!trial) {
            console.error(`Trial ${trialId} not found`);
            return;
        }
        
        trial.movements.push({
            timestamp: Date.now(),
            timeSinceTrialStart: Date.now() - trial.startTime,
            ...movementData
        });
        
        // Save periodically (every 10 movements)
        if (trial.movements.length % 10 === 0) {
            this.saveSessionData();
        }
    },
    
    /**
     * Record a user action in the current trial
     */
    recordAction(trialId, actionData) {
        if (!this.sessionData) this.loadSessionData();
        
        const trial = this.sessionData.simulatorData.trials[trialId];
        if (!trial) {
            console.error(`Trial ${trialId} not found`);
            return;
        }
        
        trial.actions.push({
            timestamp: Date.now(),
            timeSinceTrialStart: Date.now() - trial.startTime,
            ...actionData
        });
    },
    
    /**
     * Record probability distribution in the current trial
     */
    recordProbabilities(trialId, probData) {
        if (!this.sessionData) this.loadSessionData();
        
        const trial = this.sessionData.simulatorData.trials[trialId];
        if (!trial) {
            console.error(`Trial ${trialId} not found`);
            return;
        }
        
        trial.probabilities.push({
            timestamp: Date.now(),
            timeSinceTrialStart: Date.now() - trial.startTime,
            ...probData
        });
    },
    
    /**
     * Complete the current trial
     */
    completeTrial(trialId, completionData) {
        if (!this.sessionData) this.loadSessionData();
        
        const trial = this.sessionData.simulatorData.trials[trialId];
        if (!trial) {
            console.error(`Trial ${trialId} not found`);
            return;
        }
        
        trial.completed = true;
        trial.completionTime = Date.now();
        trial.duration = trial.completionTime - trial.startTime;
        trial.summary = completionData;
        
        // Update totals
        this.sessionData.simulatorData.totalTime += trial.duration;
        if (completionData.moveCount) {
            this.sessionData.simulatorData.totalMoves += completionData.moveCount;
        }
        if (completionData.totalDistance) {
            this.sessionData.simulatorData.totalDistance += completionData.totalDistance;
        }
        
        this.saveSessionData();
        console.log(`Trial ${trialId} completed:`, completionData);
    },
    
    /**
     * Record a general event
     */
    recordEvent(eventData) {
        if (!this.sessionData) this.loadSessionData();
        
        this.sessionData.events.push({
            timestamp: Date.now(),
            ...eventData
        });
        
        this.saveSessionData();
    },
    
    /**
     * Record a page visit
     */
    recordPageVisit(pageName) {
        if (!this.sessionData) this.loadSessionData();
        
        this.sessionData.pageVisits.push({
            page: pageName,
            timestamp: Date.now(),
            url: window.location.href
        });
        
        this.saveSessionData();
    },
    
    /**
     * Export all data as JSON file
     */
    exportData() {
        if (!this.sessionData) this.loadSessionData();
        
        // Add export metadata
        const exportData = {
            ...this.sessionData,
            exportTime: Date.now(),
            exportDate: new Date().toISOString()
        };
        
        // Create JSON file
        const dataStr = JSON.stringify(exportData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = `study_data_${this.userId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Data exported:', exportData);
        return exportData;
    },
    
    /**
     * Get current session data
     */
    getData() {
        if (!this.sessionData) this.loadSessionData();
        return this.sessionData;
    },
    
    /**
     * Clear all data (for testing)
     */
    clearData() {
        if (this.userId) {
            localStorage.removeItem(`study_data_${this.userId}`);
            localStorage.removeItem('studyUserId');
        }
        this.sessionData = null;
        this.userId = null;
        console.log('All data cleared');
    },
    
    /**
     * Get summary statistics
     */
    getSummary() {
        if (!this.sessionData) this.loadSessionData();
        
        const data = this.sessionData;
        return {
            userId: data.userId,
            sessionDuration: Date.now() - data.sessionStartTime,
            totalTrials: data.simulatorData.trials.length,
            completedTrials: data.simulatorData.trials.filter(t => t.completed).length,
            totalMoves: data.simulatorData.totalMoves,
            totalDistance: data.simulatorData.totalDistance,
            condition: data.condition,
            totalEvents: data.events.length,
            pagesVisited: data.pageVisits.length,
            hasDemographics: Object.keys(data.demographics).length > 0,
            hasPreSurvey: Object.keys(data.surveyResponses.preSurvey).length > 0,
            hasPostSurvey: Object.keys(data.surveyResponses.postSurvey).length > 0
        };
    }
};

// Auto-load session data when script loads (only if user ID exists)
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        const userId = localStorage.getItem('studyUserId');
        if (userId) {
            DataCollector.loadSessionData();
        }
    });
}

// Make available globally
if (typeof window !== 'undefined') {
    window.DataCollector = DataCollector;
}