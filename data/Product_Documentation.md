# AcmeCloud — Product Documentation

## 1. Overview
AcmeCloud is a SaaS platform for document collaboration and workflow automation.
It offers real-time editing, role-based access control, and an API for
integrations.

## 2. Plans & Limits
| Plan        | Price (per user/month) | Storage | API Rate Limit       |
|-------------|------------------------|---------|----------------------|
| Free        | $0                     | 2 GB    | 60 requests/minute   |
| Pro         | $12                    | 100 GB  | 600 requests/minute  |
| Enterprise  | Custom                 | Unlimited | 6,000 requests/minute |

The **Free plan supports up to 3 users**. The Pro plan has no user cap.

## 3. Authentication
The API uses **OAuth 2.0 bearer tokens**. Tokens expire after **60 minutes** and
must be refreshed using the refresh token, which is valid for 30 days. API keys
for server-to-server use can be generated from **Settings → Developer → API Keys**.

## 4. Supported File Types
AcmeCloud supports PDF, DOCX, XLSX, PPTX, PNG, and JPG. The maximum file size for
uploads on the Pro plan is **5 GB per file**.

## 5. Data Backup
Customer data is backed up **every 6 hours** and retained for 35 days.
Enterprise customers can configure custom backup schedules and point-in-time
recovery.

## 6. Service Level Agreement (SLA)
The Enterprise plan guarantees **99.9% uptime**. If uptime falls below this in a
calendar month, customers are eligible for service credits as described in the
contract.
