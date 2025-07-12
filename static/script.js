// Download medical report
function downloadReport() {
    try {
        const reportSection = document.querySelector('.report-section');
        if (!reportSection) {
            showMessage('No medical report available to download', 'error');
            return;
        }

        const reportContent = generateReportContent();
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `brain_tumor_report_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        showMessage('Report downloaded successfully', 'success');
    } catch (error) {
        console.error('Error downloading report:', error);
        showMessage('Error downloading report', 'error');
    }
}

// Print medical report
function printReport() {
    try {
        const reportSection = document.querySelector('.report-section');
        if (!reportSection) {
            showMessage('No medical report available to print', 'error');
            return;
        }

        const printWindow = window.open('', '_blank');
        const reportContent = generateReportContent();
        
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Brain Tumor Detection Report</title>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
                    
                    body { 
                        font-family: 'Inter', sans-serif; 
                        margin: 20px; 
                        line-height: 1.6;
                        color: #333;
                    }
                    
                    .header {
                        text-align: center;
                        border-bottom: 3px solid #667eea;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }
                    
                    .header h1 {
                        color: #667eea;
                        font-size: 2rem;
                        margin-bottom: 10px;
                    }
                    
                    .header .date {
                        color: #666;
                        font-size: 0.9rem;
                    }
                    
                    .section {
                        margin: 25px 0;
                        page-break-inside: avoid;
                    }
                    
                    .section h2 {
                        color: #667eea;
                        font-size: 1.3rem;
                        margin-bottom: 15px;
                        border-bottom: 2px solid #e9ecef;
                        padding-bottom: 8px;
                    }
                    
                    .diagnosis-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 15px 0;
                    }
                    
                    .diagnosis-item {
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 4px solid #667eea;
                    }
                    
                    .diagnosis-item .label {
                        font-size: 0.8rem;
                        color: #666;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 5px;
                    }
                    
                    .diagnosis-item .value {
                        font-size: 1.1rem;
                        font-weight: 600;
                        color: #333;
                    }
                    
                    .treatment-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 15px;
                        margin: 15px 0;
                    }
                    
                    .treatment-item {
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 4px solid #28a745;
                    }
                    
                    .treatment-item h3 {
                        font-size: 1rem;
                        margin-bottom: 8px;
                        color: #333;
                    }
                    
                    .precautions-list {
                        list-style: none;
                        padding: 0;
                    }
                    
                    .precautions-list li {
                        background: #fff3cd;
                        padding: 10px 15px;
                        margin-bottom: 8px;
                        border-radius: 6px;
                        border-left: 4px solid #ffc107;
                        color: #856404;
                    }
                    
                    .doctor-card {
                        background: #e7f3ff;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 4px solid #667eea;
                        color: #004085;
                        font-weight: 500;
                    }
                    
                    .footer {
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 2px solid #e9ecef;
                        text-align: center;
                        color: #666;
                        font-size: 0.9rem;
                    }
                    
                    @media print { 
                        body { margin: 0; }
                        .section { page-break-inside: avoid; }
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Brain Tumor Detection Report</h1>
                    <div class="date">Generated on: ${new Date().toLocaleDateString()}</div>
                </div>
                
                <div class="content">
                    <pre style="white-space: pre-wrap; font-family: 'Inter', sans-serif; line-height: 1.6;">${reportContent}</pre>
                </div>
                
                <div class="footer">
                    <p>Report generated by BrainScan AI</p>
                    <p>For medical use only - Consult with healthcare professionals</p>
                </div>
            </body>
            </html>
        `);
        
        printWindow.document.close();
        printWindow.print();
        
        showMessage('Print dialog opened', 'success');
    } catch (error) {
        console.error('Error printing report:', error);
        showMessage('Error printing report', 'error');
    }
}

// Generate report content for download/print
function generateReportContent() {
    const reportSection = document.querySelector('.report-section');
    if (!reportSection) return 'No report available';

    let content = 'BRAIN TUMOR DETECTION REPORT\n';
    content += '='.repeat(60) + '\n\n';
    
    // Get report date
    const reportDate = reportSection.querySelector('.date');
    if (reportDate) {
        content += `Date: ${reportDate.textContent}\n\n`;
    }
    
    // Get diagnosis details
    const diagnosisItems = reportSection.querySelectorAll('.diagnosis-item');
    if (diagnosisItems.length > 0) {
        content += 'DIAGNOSIS DETAILS\n';
        content += '-'.repeat(30) + '\n';
        diagnosisItems.forEach(item => {
            const label = item.querySelector('.label')?.textContent || '';
            const value = item.querySelector('.value')?.textContent || '';
            content += `${label}: ${value}\n`;
        });
        content += '\n';
    }
    
    // Get condition
    const condition = reportSection.querySelector('.condition');
    if (condition) {
        content += 'CONDITION\n';
        content += '-'.repeat(30) + '\n';
        content += `${condition.textContent}\n\n`;
    }
    
    // Get treatment options
    const treatmentItems = reportSection.querySelectorAll('.treatment-item');
    if (treatmentItems.length > 0) {
        content += 'TREATMENT OPTIONS\n';
        content += '-'.repeat(30) + '\n';
        treatmentItems.forEach(item => {
            const title = item.querySelector('h5')?.textContent || '';
            const description = item.querySelector('p')?.textContent || '';
            content += `${title}: ${description}\n`;
        });
        content += '\n';
    }
    
    // Get precautions
    const precautions = reportSection.querySelectorAll('.precautions-list li');
    if (precautions.length > 0) {
        content += 'PRECAUTIONS\n';
        content += '-'.repeat(30) + '\n';
        precautions.forEach(item => {
            const text = item.textContent.trim();
            content += `• ${text}\n`;
        });
        content += '\n';
    }
    
    // Get doctor recommendation
    const doctorCard = reportSection.querySelector('.doctor-card span');
    if (doctorCard) {
        content += 'RECOMMENDED SPECIALIST\n';
        content += '-'.repeat(30) + '\n';
        content += `${doctorCard.textContent}\n\n`;
    }
    
    // Get AI analysis
    const analysisSection = reportSection.querySelector('.report-section:last-child p');
    if (analysisSection) {
        content += 'AI ANALYSIS\n';
        content += '-'.repeat(30) + '\n';
        content += `${analysisSection.textContent}\n\n`;
    }
    
    content += '\n' + '='.repeat(60) + '\n';
    content += 'Report generated by BrainScan AI\n';
    content += 'For medical use only - Consult with healthcare professionals\n';
    content += `Generated on: ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}\n`;
    
    return content;
}

// Show message with improved styling
function showMessage(message, type = 'info') {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas ${getMessageIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        .message {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 0;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
            max-width: 350px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(10px);
        }
        
        .message-content {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 20px;
        }
        
        .message-error {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            border: 1px solid #ff4757;
        }
        
        .message-info {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            border: 1px solid #00d2d3;
        }
        
        .message-success {
            background: linear-gradient(135deg, #26de81, #20bf6b);
            border: 1px solid #0fb9b1;
        }
        
        .message i {
            font-size: 1.1rem;
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideOutRight {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(messageDiv);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        messageDiv.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 300);
    }, 4000);
}

// Get appropriate icon for message type
function getMessageIcon(type) {
    switch(type) {
        case 'error':
            return 'fa-exclamation-circle';
        case 'success':
            return 'fa-check-circle';
        case 'info':
        default:
            return 'fa-info-circle';
    }
}

// Theme toggle functionality
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.querySelector('.theme-toggle i');
    
    if (body.classList.contains('dark-theme')) {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.className = 'fas fa-sun';
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.className = 'fas fa-moon';
        localStorage.setItem('theme', 'dark');
    }
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    const body = document.body;
    const themeToggle = document.querySelector('.theme-toggle i');
    
    if (savedTheme === 'light') {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.className = 'fas fa-sun';
    } else {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.className = 'fas fa-moon';
    }
}); 