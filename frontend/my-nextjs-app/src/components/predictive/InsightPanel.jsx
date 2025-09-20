import { useDashboardStore } from "../../lib/store";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { Button } from "../ui/Button";
import { Badge } from "../ui/Badge";
import { 
  XIcon, 
  BrainIcon, 
  TrendingUpIcon, 
  AlertTriangleIcon, 
  BarChart3Icon,
  CalendarIcon,
  MapPinIcon,
  ShoppingBagIcon,
  DollarSignIcon,
  LightbulbIcon,
  StarIcon,
  ArrowRightIcon
} from "lucide-react";

const mockInsights = [
  {
    type: "opportunity",
    title: "High Demand Surge Expected",
    description: "MacBook Pro 16\" showing 125% increase in predicted demand at Downtown Store",
    impact: "high",
    confidence: 92,
    recommendation: "Increase inventory by 40% to meet projected demand and avoid stockouts",
    metrics: { revenue: "+$187,500", units: "+125", timeframe: "Next 7 days" },
    timeline: "Immediate action required",
    factors: ["Seasonal trend", "Marketing campaign impact", "Competitor analysis"]
  },
  {
    type: "risk",
    title: "Performance Decline Alert",
    description: "iPad Air demand dropping by 15% across all locations due to market saturation",
    impact: "medium",
    confidence: 78,
    recommendation: "Launch targeted promotional campaign or consider bundling with accessories",
    metrics: { revenue: "-$15,000", units: "-25", timeframe: "Next 14 days" },
    timeline: "Action needed within 48 hours",
    factors: ["Market competition", "Product lifecycle", "Customer preferences"]
  },
  {
    type: "trend",
    title: "Weekend Performance Pattern",
    description: "Electronics category showing consistent 23% spike in sales during weekends",
    impact: "medium",
    confidence: 85,
    recommendation: "Optimize weekend staffing levels and ensure adequate inventory allocation",
    metrics: { weekendBoost: "+23%", peakHours: "2-6 PM", avgIncrease: "$12,500" },
    timeline: "Implement for next weekend",
    factors: ["Consumer behavior", "Shopping patterns", "Promotional timing"]
  },
  {
    type: "opportunity",
    title: "Cross-Selling Potential",
    description: "Customers buying iPhones show 67% likelihood to purchase AirPods within 30 days",
    impact: "high",
    confidence: 89,
    recommendation: "Create bundle offers and targeted upselling campaigns",
    metrics: { conversionRate: "67%", avgBundle: "$180", potential: "+$45,000" },
    timeline: "Deploy within 1 week",
    factors: ["Purchase history", "Customer segmentation", "Product affinity"]
  }
];

const getImpactColor = (impact) => {
  switch (impact) {
    case "high": return "text-chart-red";
    case "medium": return "text-orange-500";
    case "low": return "text-chart-green";
    default: return "text-muted-foreground";
  }
};

const getImpactVariant = (impact) => {
  switch (impact) {
    case "high": return "destructive";
    case "medium": return "warning";
    case "low": return "success";
    default: return "outline";
  }
};

const getTypeIcon = (type) => {
  switch (type) {
    case "opportunity": return TrendingUpIcon;
    case "risk": return AlertTriangleIcon;
    case "trend": return BarChart3Icon;
    default: return BrainIcon;
  }
};

const getTypeColor = (type) => {
  switch (type) {
    case "opportunity": return "text-chart-green";
    case "risk": return "text-chart-red";
    case "trend": return "text-chart-blue";
    default: return "text-primary";
  }
};

export default function InsightPanel() {
  const { insightPanelOpen, setInsightPanelOpen } = useDashboardStore();

  if (!insightPanelOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
        onClick={() => setInsightPanelOpen(false)}
      />
      
      {/* Panel */}
      <div className="fixed inset-y-0 right-0 z-50 w-96 bg-background/95 backdrop-blur-xl border-l border-border shadow-2xl animate-slide-in">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border bg-card/50">
            <div className="flex items-center gap-2">
              <BrainIcon className="h-5 w-5 text-primary" />
              <div>
                <h2 className="text-lg font-semibold text-foreground">AI Insights</h2>
                <p className="text-xs text-muted-foreground">Powered by machine learning</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setInsightPanelOpen(false)}
              className="h-8 w-8 hover:bg-destructive/20 hover:text-destructive"
            >
              <XIcon className="h-4 w-4" />
            </Button>
          </div>

          {/* Summary Stats */}
          <div className="p-6 border-b border-border">
            <Card className="glass-card">
              <CardContent className="p-4">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-lg font-bold text-chart-green">{mockInsights.length}</p>
                    <p className="text-xs text-muted-foreground">Active Insights</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-chart-blue">
                      {Math.round(mockInsights.reduce((sum, i) => sum + i.confidence, 0) / mockInsights.length)}%
                    </p>
                    <p className="text-xs text-muted-foreground">Avg Confidence</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-primary">
                      {mockInsights.filter(i => i.impact === "high").length}
                    </p>
                    <p className="text-xs text-muted-foreground">High Priority</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Insights List */}
            {mockInsights.map((insight, index) => {
              const TypeIcon = getTypeIcon(insight.type);
              const typeColor = getTypeColor(insight.type);
              const impactColor = getImpactColor(insight.impact);
              const impactVariant = getImpactVariant(insight.impact);

              return (
                <Card key={index} className="glass-card hover:bg-card/90 transition-all border-l-4 border-l-primary">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`p-1.5 rounded-lg bg-background`}>
                          <TypeIcon className={`h-4 w-4 ${typeColor}`} />
                        </div>
                        <div>
                          <CardTitle className="text-sm">{insight.title}</CardTitle>
                          <p className="text-xs text-muted-foreground capitalize">{insight.type}</p>
                        </div>
                      </div>
                      <Badge variant={impactVariant} className="text-xs capitalize">
                        {insight.impact} impact
                      </Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {insight.description}
                    </p>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-2 gap-3 p-3 bg-muted/30 rounded-lg">
                      {Object.entries(insight.metrics).map(([key, value], idx) => (
                        <div key={idx} className="text-center">
                          <p className="text-sm font-semibold text-foreground">{value}</p>
                          <p className="text-xs text-muted-foreground capitalize">
                            {key.replace(/([A-Z])/g, ' $1').trim()}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* Timeline */}
                    <div className="flex items-center gap-2 p-2 bg-primary/10 rounded-lg">
                      <CalendarIcon className="h-4 w-4 text-primary" />
                      <div>
                        <p className="text-xs font-medium text-primary">Timeline</p>
                        <p className="text-xs text-foreground">{insight.timeline}</p>
                      </div>
                    </div>

                    {/* Confidence */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground flex items-center gap-1">
                          <StarIcon className="h-3 w-3" />
                          Confidence Level
                        </span>
                        <span className="text-xs font-medium">{insight.confidence}%</span>
                      </div>
                      <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-primary to-chart-green transition-all duration-1000"
                          style={{ width: `${insight.confidence}%` }}
                        />
                      </div>
                    </div>

                    {/* Contributing Factors */}
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-foreground flex items-center gap-1">
                        <BarChart3Icon className="h-3 w-3" />
                        Key Factors
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {insight.factors.map((factor, idx) => (
                          <Badge key={idx} variant="outline" className="text-xs">
                            {factor}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Recommendation */}
                    <div className="p-3 bg-gradient-to-r from-primary/10 to-chart-green/10 rounded-lg border border-primary/20">
                      <div className="flex items-start gap-2">
                        <LightbulbIcon className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-xs font-medium text-primary mb-1">Recommended Action</p>
                          <p className="text-xs text-foreground leading-relaxed">{insight.recommendation}</p>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" className="flex-1 text-xs">
                        View Details
                      </Button>
                      <Button variant="default" size="sm" className="flex-1 text-xs">
                        <span>Take Action</span>
                        <ArrowRightIcon className="h-3 w-3 ml-1" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}

            {/* Performance Metrics */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <BarChart3Icon className="h-4 w-4 text-primary" />
                  Model Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Prediction Accuracy</span>
                    <span className="text-chart-green font-medium">94.2%</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Data Points Analyzed</span>
                    <span className="text-foreground font-medium">2.3M</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Last Model Update</span>
                    <span className="text-foreground">2 hours ago</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Processing Speed</span>
                    <span className="text-chart-blue font-medium">1.2ms avg</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Footer */}
          <div className="border-t border-border p-4 bg-card/30">
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                className="flex-1 text-xs"
                onClick={() => setInsightPanelOpen(false)}
              >
                Close Panel
              </Button>
              <Button 
                variant="default" 
                className="flex-1 text-xs"
              >
                Export Report
              </Button>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-2">
              Insights updated in real-time
            </p>
          </div>
        </div>
      </div>
    </>
  );
}