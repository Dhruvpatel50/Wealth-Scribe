'use client'
import Link from 'next/link';
import React from 'react';
import { useAuth, UserButton } from '@clerk/nextjs';
import Image from 'next/image';

export default function Home() {
  const { isSignedIn } = useAuth();

  return (
    <div className="min-h-screen bg-[#212121] text-[#EDEDED]">
      <header className="bg-[#212121] shadow-md sticky top-0 z-10">
        <div className="container mx-auto px-4 sm:px-6 py-4 flex justify-between items-center">
          <Link href={'/'}>
            <h1 className="text-lg sm:text-2xl font-bold text-blue-400">Wealth-Scribe</h1>
          </Link>
          <nav className="hidden sm:flex space-x-6 items-center">
            <Link href={"#features"} className="text-[#EDEDED] hover:text-blue-400">Features</Link>
            <Link href={"#how-it-works"} className="text-[#EDEDED] hover:text-blue-400">How It Works</Link>
            <Link href={"#contact"} className="text-[#EDEDED] hover:text-blue-400">Contact</Link>
            {isSignedIn ? (
              <UserButton />
            ) : (
              <Link
                href={'/sign-in'}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700"
              >
                Get Started
              </Link>
            )}
          </nav>
          <button className="sm:hidden text-blue-400">Menu</button>
        </div>
      </header>

      <section className="bg-[#212121] py-16 sm:py-20">
        <div className="container mx-auto px-4 sm:px-6 text-center">
          <h2 className="text-3xl sm:text-4xl font-extrabold text-blue-400 mb-4">
            Unlock Accurate Insights from Financial Reports
          </h2>
          <p className="text-sm sm:text-lg text-[#EDEDED] mb-6">
            Transform complex financial PDFs into actionable data and stunning visualizations.
          </p>
          <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            {isSignedIn ? (
              <Link
                href={'/dashboard'}
                className="px-4 sm:px-6 py-2 sm:py-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700"
              >
                Go to Dashboard
              </Link>
            ) : (
              <Link
                href={'/sign-in'}
                className="px-4 sm:px-6 py-2 sm:py-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700"
              >
                Get Started
              </Link>
            )}
            <Link
              href={'/dashboard'}
              className="px-4 sm:px-6 py-2 sm:py-3 bg-[#333333] text-blue-400 rounded-lg shadow-md hover:bg-[#444444]"
            >
              Upload a Report
            </Link>
          </div>
          <div className="mt-8 sm:mt-10">
            <Image
              src={'https://cdn.prod.website-files.com/66edcf86968ea143cdc05871/66edcf86968ea143cdc058b1_home-hero-p-800.png'}
              alt="error"
              width={500}
              height={100}
              className="mx-auto rounded-lg shadow-lg"
            />
          </div>
        </div>
      </section>

      {/* Features, How It Works, and Footer remain the same */}
      <section id="features" className="py-16 sm:py-20 bg-[#212121]">
        <div className="container mx-auto px-4 sm:px-6">
          <h3 className="text-2xl sm:text-3xl font-bold text-center text-[#EDEDED] mb-8 sm:mb-12">
            Why Choose Wealth-Scribe?
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
            {[
              { title: "100% Accurate", desc: "Ensure reliable and precise data from PDFs." },
              { title: "Speedy Processing", desc: "Get insights in seconds." },
              { title: "Custom Visualizations", desc: "Interactive charts and graphs." },
              { title: "AI-Driven Insights", desc: "Powered by cutting-edge AI." },
            ].map((feature, idx) => (
              <div
                key={idx}
                className="bg-[#333333] p-4 sm:p-6 rounded-lg shadow-md text-center hover:shadow-lg"
              >
                <h4 className="text-lg sm:text-xl font-semibold text-blue-400 mb-2">{feature.title}</h4>
                <p className="text-sm sm:text-base text-[#EDEDED]">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="how-it-works" className="bg-[#212121] py-16 sm:py-20">
        <div className="container mx-auto px-4 sm:px-6 text-center">
          <h3 className="text-2xl sm:text-3xl font-bold text-[#EDEDED] mb-8 sm:mb-12">How It Works</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 sm:gap-8">
            {[
              {
                step: "1",
                title: "Upload a PDF",
                desc: "Drag and drop or securely upload your file.",
              },
              {
                step: "2",
                title: "Extract Metrics",
                desc: "Automatically retrieve revenue, profit, and more.",
              },
              {
                step: "3",
                title: "Visualize Data",
                desc: "Interactive charts and JSON-ready data.",
              },
            ].map((item, idx) => (
              <div
                key={idx}
                className="bg-[#333333] p-4 sm:p-6 rounded-lg shadow-md hover:shadow-lg text-center"
              >
                <div className="text-3xl sm:text-4xl font-bold text-blue-400 mb-4">{item.step}</div>
                <h4 className="text-lg sm:text-xl font-semibold text-[#EDEDED] mb-2">{item.title}</h4>
                <p className="text-sm sm:text-base text-[#EDEDED]">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer id="contact" className="bg-[#212121] py-6">
        <div className="container mx-auto px-4 sm:px-6 text-center text-[#EDEDED]">
          <p>&copy; 2025 Wealth-Scribe. All Rights Reserved.</p>
          <div className="flex justify-center space-x-4 sm:space-x-6 mt-4">
            <a href="#" className="text-sm sm:text-base hover:underline">Privacy Policy</a>
            <a href="#" className="text-sm sm:text-base hover:underline">Terms of Service</a>
            <a href="#" className="text-sm sm:text-base hover:underline">Contact Us</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
