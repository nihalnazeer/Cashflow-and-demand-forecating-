import { useDashboardStore } from "../../lib/store";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { 
  TrendingUpIcon, 
  TrendingDownIcon, 
  MinusIcon, 
  BrainIcon, 
  EyeIcon, 
  DownloadIcon,
  AlertCircleIcon,
  CheckCircleIcon 
} from "lucide-react";

const getTrendIcon = (trend) => {
  switch (trend) {
    case "up": return TrendingUpIcon;
    case "down": return TrendingDownIcon;
    default: return MinusIcon;
  }
};

const getTrendColor = (trend) => {
  switch (trend) {
    case "up": return "text-chart-green";
    case "down": return "text-chart-red";
    default: return "text-muted-foreground";
  }
};

const getConfidenceVariant = (confidence) => {
  if (confidence >= 85) return "success";
  if (confidence >= 70) return "warning";
  return "destructive";
};

const getConfidenceIcon = (confidence) => {
  if (confidence >= 85) return CheckCircleIcon;
  return AlertCircleIcon;
};

const formatCurrency = (value) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

export default function ForecastTable() {
  const { forecastData, setInsightPanelOpen } = useDashboardStore();

  const totalPredictedRevenue = forecastData.reduce((sum, item) => sum + item.predictedSales, 0);
  const totalPredictedDemand = forecastData.reduce((sum, item) => sum + item.predictedDemand, 0);
  const averageConfidence = forecastData.reduce((sum, item) => sum + item.confidence, 0) / forecastData.length;

  const handleViewInsights = (item) => {
    setInsightPanelOpen(true);
    // In a real app, you would pass the specific item data to the insight panel
  };

  const handleExportForecast = () => {
    const csvContent = [
      ["Shop", "Item", "Predicted Demand", "Predicted Sales", "Confidence", "Trend"],
      ...forecastData.map(item => [
        item.shop,
        item.item,
        item.predictedDemand,
        item.predictedSales,
        `${item.confidence}%`,
        item.trend
      ])
    ].map(row => row.join(",")).join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sales-forecast.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <Card className="col-span-full animate-slide-in">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BrainIcon className="h-5 w-5 text-primary" />
              AI Sales Forecast
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Machine learning powered demand predictions for the next 30 days
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {averageConfidence.toFixed(0)}% Avg Confidence
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportForecast}
              className="flex items-center gap-2"
            >
              <DownloadIcon className="h-4 w-4" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="glass-card p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-chart-green">
              {formatCurrency(totalPredictedRevenue)}
            </p>
            <p className="text-sm text-muted-foreground">Predicted Revenue</p>
          </div>
          <div className="glass-card p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-chart-blue">
              {totalPredictedDemand.toLocaleString()}
            </p>
            <p className="text-sm text-muted-foreground">Predicted Demand</p>
          </div>
          <div className="glass-card p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-chart-purple">
              {averageConfidence.toFixed(1)}%
            </p>
            <p className="text-sm text-muted-foreground">Confidence Level</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Shop Location</th>
                <th className="text-left p-4 text-sm font-medium text-muted-foreground">Product Item</th>
                <th className="text-right p-4 text-sm font-medium text-muted-foreground">Predicted Demand</th>
                <th className="text-right p-4 text-sm font-medium text-muted-foreground">Predicted Sales</th>
                <th className="text-center p-4 text-sm font-medium text-muted-foreground">Trend</th>
                <th className="text-center p-4 text-sm font-medium text-muted-foreground">Confidence</th>
                <th className="text-center p-4 text-sm font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {forecastData.map((item, index) => {
                const TrendIcon = getTrendIcon(item.trend);
                const trendColor = getTrendColor(item.trend);
                const confidenceVariant = getConfidenceVariant(item.confidence);
                const ConfidenceIcon = getConfidenceIcon(item.confidence);
                
                return (
                  <tr key={index} className="border-b border-border hover:bg-muted/30 transition-colors">
                    <td className="p-4">
                      <div className="space-y-1">
                        <p className="font-medium text-foreground">
                          {item.shop}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Location ID: {(index + 1).toString().padStart(3, '0')}
                        </p>
                      </div>
                    </td>
                    
                    <td className="p-4">
                      <span className="font-medium text-foreground">
                        {item.item}
                      </span>
                    </td>
                    
                    <td className="p-4 text-right">
                      <div className="space-y-1">
                        <span className="font-semibold text-chart-blue">
                          {item.predictedDemand.toLocaleString()}
                        </span>
                        <p className="text-xs text-muted-foreground">units</p>
                      </div>
                    </td>
                    
                    <td className="p-4 text-right">
                      <div className="space-y-1">
                        <span className="font-semibold text-chart-green">
                          {formatCurrency(item.predictedSales)}
                        </span>
                        <p className="text-xs text-muted-foreground">
                          ${(item.predictedSales / item.predictedDemand).toFixed(0)} avg
                        </p>
                      </div>
                    </td>
                    
                    <td className="p-4 text-center">
                      <div className="flex items-center justify-center gap-1">
                        <TrendIcon className={`h-4 w-4 ${trendColor}`} />
                        <span className={`text-sm font-medium capitalize ${trendColor}`}>
                          {item.trend}
                        </span>
                      </div>
                    </td>
                    
                    <td className="p-4 text-center">
                      <div className="flex items-center justify-center gap-2">
                        <ConfidenceIcon className={`h-4 w-4 ${
                          item.confidence >= 85 ? 'text-chart-green' : 'text-orange-500'
                        }`} />
                        <Badge variant={confidenceVariant} className="text-xs">
                          {item.confidence}%
                        </Badge>
                      </div>
                    </td>
                    
                    <td className="p-4 text-center">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleViewInsights(item)}
                        className="flex items-center gap-1 text-xs"
                      >
                        <EyeIcon className="h-3 w-3" />
                        Insights
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Forecast Insights */}
        <div className="mt-6 p-4 glass-card rounded-lg">
          <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
            <BrainIcon className="h-4 w-4 text-primary" />
            Key Forecast Insights
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
            <div className="space-y-1">
              <p className="text-chart-green font-medium">↗ High Growth Products</p>
              <p className="text-muted-foreground">MacBook Pro 16" and iPhone 15 Pro showing strong upward trends</p>
            </div>
            <div className="space-y-1">
              <p className="text-chart-blue font-medium">📍 Top Performing Location</p>
              <p className="text-muted-foreground">Downtown Store expected to generate highest revenue</p>
            </div>
            <div className="space-y-1">
              <p className="text-orange-500 font-medium">⚠ Attention Needed</p>
              <p className="text-muted-foreground">iPad Air showing declining trend, consider promotional strategies</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}