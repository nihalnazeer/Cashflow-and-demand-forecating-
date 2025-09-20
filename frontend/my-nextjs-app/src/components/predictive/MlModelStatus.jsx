import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { 
  BrainIcon, 
  CheckCircleIcon, 
  ClockIcon, 
  AlertCircleIcon,
  RefreshCwIcon,
  TrendingUpIcon
} from "lucide-react";

const modelMetrics = [
  { name: "Revenue Forecasting", accuracy: 94.2, status: "active", lastTrained: "2 hours ago" },
  { name: "Demand Prediction", accuracy: 89.7, status: "active", lastTrained: "4 hours ago" },
  { name: "Customer Segmentation", accuracy: 92.1, status: "training", lastTrained: "1 day ago" },
  { name: "Price Optimization", accuracy: 87.3, status: "pending", lastTrained: "3 days ago" }
];

const getStatusIcon = (status) => {
  switch (status) {
    case "active": return CheckCircleIcon;
    case "training": return ClockIcon;
    case "pending": return AlertCircleIcon;
    default: return BrainIcon;
  }
};

const getStatusColor = (status) => {
  switch (status) {
    case "active": return "text-chart-green";
    case "training": return "text-chart-blue";
    case "pending": return "text-orange-500";
    default: return "text-muted-foreground";
  }
};

const getStatusVariant = (status) => {
  switch (status) {
    case "active": return "success";
    case "training": return "outline";
    case "pending": return "warning";
    default: return "secondary";
  }
};

export default function MLModelStatus() {
  return (
    <Card className="animate-slide-in">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BrainIcon className="h-5 w-5 text-primary" />
            ML Model Status
          </CardTitle>
          <Button variant="outline" size="sm" className="flex items-center gap-1">
            <RefreshCwIcon className="h-3 w-3" />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {modelMetrics.map((model, index) => {
          const StatusIcon = getStatusIcon(model.status);
          const statusColor = getStatusColor(model.status);
          const statusVariant = getStatusVariant(model.status);
          
          return (
            <div key={index} className="p-4 glass-card rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <StatusIcon className={`h-4 w-4 ${statusColor}`} />
                  <span className="font-medium text-foreground">{model.name}</span>
                </div>
                <Badge variant={statusVariant} className="text-xs capitalize">
                  {model.status}
                </Badge>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <TrendingUpIcon className="h-3 w-3 text-chart-green" />
                    <span className="text-muted-foreground">Accuracy:</span>
                    <span className="font-medium text-chart-green">{model.accuracy}%</span>
                  </div>
                </div>
                <span className="text-xs text-muted-foreground">
                  Updated {model.lastTrained}
                </span>
              </div>
              
              {/* Accuracy Bar */}
              <div className="mt-2">
                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-chart-green transition-all duration-1000"
                    style={{ width: `${model.accuracy}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
        
        <div className="pt-4 border-t border-border">
          <div className="grid grid-cols-2 gap-4 text-center text-sm">
            <div>
              <p className="text-lg font-bold text-chart-green">91.1%</p>
              <p className="text-xs text-muted-foreground">Overall Accuracy</p>
            </div>
            <div>
              <p className="text-lg font-bold text-chart-blue">2.3M</p>
              <p className="text-xs text-muted-foreground">Training Samples</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}