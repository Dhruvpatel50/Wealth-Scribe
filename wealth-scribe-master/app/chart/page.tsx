"use client";
import { TrendingUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { useEffect, useState } from "react";

import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import {
    ChartConfig,
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart";

interface ChartData {
    metric: string;
    value: number;
}

const defaultData = [
    { metric: "Revenue", value: 0 },
    { metric: "PAT", value: 0 },
    { metric: "Net", value: 0 },
];

const chartConfig = {
    value: {
        label: `Amount (₹)`,
        color: "hsl(var(--chart-1))",
    },
} satisfies ChartConfig;

const FinancialChart = ({ title, data, color }: { title: string; data: ChartData[]; color: string }) => {
    return (
        <Card className="w-full bg-[#212121] text-white border border-[#696969]">
            <CardHeader>
                <CardTitle className="text-white">{title}</CardTitle>
                <CardDescription className="text-gray-400">Revenue, PAT, and Net Profit</CardDescription>
            </CardHeader>
            <CardContent>
                <ChartContainer config={chartConfig}>
                    <BarChart accessibilityLayer data={data}>
                        <CartesianGrid vertical={false} stroke="#444" />
                        <XAxis
                            dataKey="metric"
                            tickLine={false}
                            tickMargin={10}
                            axisLine={false}
                            tickFormatter={(value) => value.slice(0, 3)}
                        />
                        <YAxis
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(value) => value.toFixed(0)}
                            domain={['auto', 'auto']}
                        />
                        <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                        <Bar
                            dataKey="value"
                            fill={color}
                            radius={[4, 4, 0, 0]}
                            minPointSize={15}
                        />
                    </BarChart>
                </ChartContainer>
            </CardContent>
            <CardFooter className="flex-col items-start gap-2 text-sm text-gray-300">
                <div className="flex gap-2 font-medium leading-none">
                    Trending up by 5.2% this quarter <TrendingUp className="h-4 w-4 text-white" />
                </div>
                <div className="leading-none">Financial performance for this quarter</div>
            </CardFooter>
        </Card>
    );
};

const parseData = (urlData: string) => {
    try {
        const parsedData = decodeURIComponent(urlData);
        return JSON.parse(parsedData);
    } catch (error) {
        console.error("Error parsing data:", error);
        return {};
    }
};

export default function ChartComponent() {
    const [chartData, setChartData] = useState<any>({ Q1: defaultData, Q2: defaultData });

    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const data = urlParams.get("data");
        if (data) {
            const parsedData = parseData(data);
            const q1Data = parsedData["Q1 FY2023-24"] || {};
            const q2Data = parsedData["Q2 FY2023-24"] || {};
            setChartData({
                Q1: [
                    { metric: "Revenue", value: q1Data["Revenue"] || 0 },
                    { metric: "PAT", value: q1Data["PBT"] || 0 },
                    { metric: "Net", value: q1Data["Net Profit"] || 0 },
                ],
                Q2: [
                    { metric: "Revenue", value: q2Data["Revenue"] || 0 },
                    { metric: "PAT", value: q2Data["PBT"] || 0 },
                    { metric: "Net", value: q2Data["Net Profit"] || 0 },
                ],
            });
        } else {
            setChartData({ Q1: defaultData, Q2: defaultData });
        }
    }, []);

    return (
        <div className="w-full bg-[#212121] p-4">
            {/* Desktop: Side-by-side charts */}
            <div className="hidden md:flex flex-wrap justify-center gap-6">
                <div className="w-1/2">
                    <FinancialChart title="Q1 Financials" data={chartData.Q1} color="hsl(var(--chart-1))" />
                </div>
                <div className="w-1/2">
                    <FinancialChart title="Q2 Financials" data={chartData.Q2} color="hsl(var(--chart-2))" />
                </div>
            </div>

            {/* Mobile: One chart at a time */}
            <div className="md:hidden space-y-6">
                <FinancialChart title="Q1 Financials" data={chartData.Q1} color="hsl(var(--chart-1))" />
                <FinancialChart title="Q2 Financials" data={chartData.Q2} color="hsl(var(--chart-2))" />
            </div>
        </div>
    );
}
